"""Extract and clean page-level text from the project's PDF corpus.

The output is JSON Lines (one JSON object per retained PDF page) so later
chunking and retrieval steps can preserve exact source-page metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


MIN_PAGE_CHARACTERS = 30
EDGE_LINE_WINDOW = 5
REPEATED_LINE_RATIO = 0.30


def normalize_line(line: str) -> str:
    """Normalize a line for repeated-header/footer comparison."""
    return re.sub(r"\s+", " ", line).strip()


def candidate_edge_lines(page_text: str) -> set[str]:
    """Return short non-empty lines near the top and bottom of a page."""
    lines = [normalize_line(line) for line in page_text.splitlines()]
    lines = [line for line in lines if line]
    edge_lines = lines[:EDGE_LINE_WINDOW] + lines[-EDGE_LINE_WINDOW:]
    return {line for line in edge_lines if 3 <= len(line) <= 200}


def find_repeated_edge_lines(page_texts: list[str]) -> set[str]:
    """Identify page furniture repeated near document edges."""
    populated_pages = [text for text in page_texts if len(text.strip()) >= MIN_PAGE_CHARACTERS]
    if not populated_pages:
        return set()

    counts: Counter[str] = Counter()
    for text in populated_pages:
        counts.update(candidate_edge_lines(text))

    threshold = max(3, round(len(populated_pages) * REPEATED_LINE_RATIO))
    return {line for line, count in counts.items() if count >= threshold}


def clean_page_text(raw_text: str, repeated_lines: set[str]) -> str:
    """Remove page furniture and normalize PDF line wrapping."""
    retained_lines: list[str] = []

    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line or line in repeated_lines:
            continue

        # This running header is common in the selected NIST publications.
        if line.lower().startswith("this publication is available free of charge from:"):
            continue

        # Remove isolated PDF page numbers while retaining numbers in prose.
        if re.fullmatch(r"\d{1,3}", line):
            continue

        retained_lines.append(line)

    text = "\n".join(retained_lines)

    # Repair words split across PDF line boundaries, e.g. "manufac-\nturing".
    text = re.sub(r"(?<=[A-Za-z])[-‐]\n(?=[a-z])", "", text)

    # Join ordinary wrapped lines while retaining likely headings and list items.
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z(])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(pdf_path: Path) -> tuple[list[dict], dict]:
    """Extract one PDF and return retained page records and diagnostics."""
    reader = PdfReader(pdf_path)
    raw_pages = [(page.extract_text() or "") for page in reader.pages]
    repeated_lines = find_repeated_edge_lines(raw_pages)

    records: list[dict] = []
    skipped_pages: list[int] = []

    for page_number, raw_text in enumerate(raw_pages, start=1):
        cleaned_text = clean_page_text(raw_text, repeated_lines)
        if len(cleaned_text) < MIN_PAGE_CHARACTERS:
            skipped_pages.append(page_number)
            continue

        records.append(
            {
                "document": pdf_path.stem,
                "filename": pdf_path.name,
                "pdf_page": page_number,
                "raw_text": raw_text.strip(),
                "cleaned_text": cleaned_text,
                "cleaned_character_count": len(cleaned_text),
            }
        )

    report = {
        "document": pdf_path.stem,
        "filename": pdf_path.name,
        "total_pdf_pages": len(raw_pages),
        "retained_pages": len(records),
        "skipped_pages": skipped_pages,
        "repeated_edge_lines_removed": sorted(repeated_lines),
        "replacement_characters_in_raw_text": sum(text.count("\ufffd") for text in raw_pages),
    }
    return records, report


def extract_corpus(input_dir: Path, output_file: Path, report_file: Path) -> None:
    """Extract every PDF in input_dir into one page-level JSONL corpus."""
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    all_records: list[dict] = []
    reports: list[dict] = []
    for pdf_path in pdf_paths:
        records, report = extract_document(pdf_path)
        all_records.extend(records)
        reports.append(report)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "documents_processed": len(pdf_paths),
        "total_retained_pages": len(all_records),
        "documents": reports,
    }
    report_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Processed {len(pdf_paths)} documents")
    print(f"Retained {len(all_records)} pages")
    print(f"Wrote page corpus to {output_file}")
    print(f"Wrote extraction report to {report_file}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory containing source PDFs",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=project_root / "data" / "processed" / "pages.jsonl",
        help="Destination JSONL file",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=project_root / "data" / "processed" / "extraction_report.json",
        help="Destination extraction diagnostics file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    extract_corpus(arguments.input_dir, arguments.output_file, arguments.report_file)
