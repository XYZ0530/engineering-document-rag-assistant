"""Create token-aware, source-preserving chunks from cleaned page records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256

# Both configurations stay below the model's 256-token truncation limit.
CHUNK_CONFIGS = (
    {"target_tokens": 128, "overlap_tokens": 24},
    {"target_tokens": 220, "overlap_tokens": 40},
)


def load_tokenizer(
    model_name: str = MODEL_NAME,
    local_tokenizer_dir: Path | None = None,
) -> PreTrainedTokenizerBase:
    """Load a local tokenizer when available, otherwise fetch it by model name."""
    if local_tokenizer_dir and local_tokenizer_dir.exists():
        return AutoTokenizer.from_pretrained(local_tokenizer_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count model input tokens, including special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def split_sentence_units(text: str) -> list[str]:
    """Split page text into lightweight paragraph/sentence units.

    Newlines from the cleaner preserve headings and list structure. Within each
    line, a conservative regular expression splits after sentence-ending
    punctuation when the next segment appears to start a new sentence.
    """
    units: list[str] = []
    boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])")

    for block in re.split(r"\n+", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        units.extend(part.strip() for part in boundary.split(block) if part.strip())

    return units


def split_oversized_unit(
    unit: str,
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
) -> list[str]:
    """Token-split an unusually long sentence/table that cannot fit a chunk."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_budget = target_tokens - special_tokens
    token_ids = tokenizer.encode(unit, add_special_tokens=False, truncation=False)

    if len(token_ids) <= content_budget:
        return [unit]

    pieces = []
    for start in range(0, len(token_ids), content_budget):
        piece = tokenizer.decode(
            token_ids[start : start + content_budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if piece:
            pieces.append(piece)
    return pieces


def prepare_units(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
) -> list[str]:
    """Create units and guarantee that each can fit within the token budget."""
    prepared: list[str] = []
    for unit in split_sentence_units(text):
        prepared.extend(split_oversized_unit(unit, tokenizer, target_tokens))
    return prepared


def choose_overlap_start(
    units: list[str],
    chunk_start: int,
    chunk_end: int,
    overlap_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Select complete trailing units for the next chunk's overlap."""
    overlap_start = chunk_end
    for candidate_start in range(chunk_end - 1, chunk_start - 1, -1):
        candidate = "\n".join(units[candidate_start:chunk_end])
        if count_tokens(candidate, tokenizer) > overlap_tokens:
            break
        overlap_start = candidate_start

    if overlap_start < chunk_end and overlap_start > chunk_start:
        return overlap_start

    # A sentence can be larger than the desired overlap budget. When the chunk
    # contains multiple units, allow one bounded trailing unit as a fallback so
    # overlap does not silently disappear. The 2x cap prevents a long sentence
    # from consuming almost the entire next chunk.
    if chunk_end - chunk_start > 1:
        last_unit = units[chunk_end - 1]
        last_unit_tokens = count_tokens(last_unit, tokenizer)
        if last_unit_tokens <= overlap_tokens * 2:
            return chunk_end - 1

    # Progress is mandatory when one large unit consumes the whole chunk.
    return chunk_end


def count_adjacent_token_overlap(
    first_text: str,
    second_text: str,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """Count matching suffix/prefix content tokens in adjacent chunks."""
    first_ids = tokenizer.encode(
        first_text, add_special_tokens=False, truncation=False
    )
    second_ids = tokenizer.encode(
        second_text, add_special_tokens=False, truncation=False
    )
    maximum = min(len(first_ids), len(second_ids))
    for size in range(maximum, 0, -1):
        if first_ids[-size:] == second_ids[:size]:
            return size
    return 0


def chunk_page(
    record: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Chunk one page without crossing its source-page boundary."""
    if overlap_tokens <= 0 or overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be positive and smaller than target_tokens")

    units = prepare_units(record["cleaned_text"], tokenizer, target_tokens)
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(units):
        end = start
        best_text = ""

        while end < len(units):
            candidate = "\n".join(units[start : end + 1])
            if count_tokens(candidate, tokenizer) > target_tokens:
                break
            best_text = candidate
            end += 1

        # Oversized units were pre-split, so this indicates a programming error.
        if end == start or not best_text:
            raise RuntimeError(
                f"Unable to fit a unit in {target_tokens} tokens for "
                f"{record['document']} page {record['pdf_page']}"
            )

        chunk_number = len(chunks) + 1
        chunk_id = (
            f"{record['document']}-p{record['pdf_page']:03d}"
            f"-t{target_tokens}-c{chunk_number:03d}"
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document": record["document"],
                "filename": record["filename"],
                "pdf_page": record["pdf_page"],
                "chunk_index_in_page": chunk_number,
                "target_tokens": target_tokens,
                "overlap_target_tokens": overlap_tokens,
                "token_count": count_tokens(best_text, tokenizer),
                "word_count": len(best_text.split()),
                "text": best_text,
            }
        )

        if end >= len(units):
            break
        start = choose_overlap_start(
            units, start, end, overlap_tokens, tokenizer
        )

    return chunks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_chunk_corpus(
    page_records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in page_records:
        chunks.extend(
            chunk_page(record, tokenizer, target_tokens, overlap_tokens)
        )
    return chunks


def validate_chunks(
    chunks: list[dict[str, Any]],
    page_records: list[dict[str, Any]],
    target_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int]:
    """Validate the invariants needed before embedding."""
    ids = [chunk["chunk_id"] for chunk in chunks]
    source_pages = {(record["document"], record["pdf_page"]) for record in page_records}
    chunk_pages = {(chunk["document"], chunk["pdf_page"]) for chunk in chunks}

    adjacent_overlaps = [
        count_adjacent_token_overlap(first["text"], second["text"], tokenizer)
        for first, second in zip(chunks, chunks[1:])
        if first["document"] == second["document"]
        and first["pdf_page"] == second["pdf_page"]
    ]

    report = {
        "chunks": len(chunks),
        "duplicate_chunk_ids": len(ids) - len(set(ids)),
        "empty_chunks": sum(not chunk["text"].strip() for chunk in chunks),
        "over_budget_chunks": sum(chunk["token_count"] > target_tokens for chunk in chunks),
        "missing_source_pages": len(source_pages - chunk_pages),
        "unknown_source_pages": len(chunk_pages - source_pages),
        "adjacent_same_page_pairs": len(adjacent_overlaps),
        "zero_overlap_pairs": sum(size == 0 for size in adjacent_overlaps),
        "median_actual_overlap_tokens": (
            sorted(adjacent_overlaps)[len(adjacent_overlaps) // 2]
            if adjacent_overlaps
            else 0
        ),
        "maximum_actual_overlap_tokens": max(adjacent_overlaps, default=0),
    }
    return report


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages-file",
        type=Path,
        default=project_root / "data" / "processed" / "pages.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--local-tokenizer-dir",
        type=Path,
        default=project_root / "models" / "all-MiniLM-L6-v2-tokenizer",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    pages = read_jsonl(arguments.pages_file)
    model_tokenizer = load_tokenizer(local_tokenizer_dir=arguments.local_tokenizer_dir)

    for config in CHUNK_CONFIGS:
        target = config["target_tokens"]
        overlap = config["overlap_tokens"]
        corpus = build_chunk_corpus(pages, model_tokenizer, target, overlap)
        output_path = arguments.output_dir / f"chunks_{target}.jsonl"
        write_jsonl(corpus, output_path)
        report = validate_chunks(corpus, pages, target, model_tokenizer)
        print(f"{target}-token configuration: {report}")
        print(f"Wrote {output_path}")
