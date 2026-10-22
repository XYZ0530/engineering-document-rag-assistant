"""Build sentence embeddings and perform exact semantic retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding_model(
    local_model_dir: Path | None = None,
    model_name: str = MODEL_NAME,
    device: str = "cpu",
) -> SentenceTransformer:
    """Load a local Sentence Transformer when available."""
    source = str(local_model_dir) if local_model_dir and local_model_dir.exists() else model_name
    return SentenceTransformer(source, device=device)


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict[str, Any]],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode chunk text as normalized float32 vectors."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def validate_embeddings(
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    expected_dimension: int = EXPECTED_DIMENSION,
) -> dict[str, Any]:
    """Validate matrix/metadata alignment and vector numerical properties."""
    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([])
    return {
        "chunk_records": len(chunks),
        "embedding_rows": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "row_alignment_valid": embeddings.ndim == 2 and embeddings.shape[0] == len(chunks),
        "dimension_valid": embeddings.ndim == 2 and embeddings.shape[1] == expected_dimension,
        "finite_values": bool(np.isfinite(embeddings).all()),
        "minimum_norm": float(norms.min()) if norms.size else 0.0,
        "maximum_norm": float(norms.max()) if norms.size else 0.0,
        "normalized": bool(np.allclose(norms, 1.0, atol=1e-5)) if norms.size else False,
    }


def save_embedding_artifact(
    embeddings: np.ndarray,
    output_path: Path,
    chunk_file: Path,
    chunks: list[dict[str, Any]],
    model_name: str = MODEL_NAME,
) -> Path:
    """Save vectors plus a manifest binding rows to the source chunk file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "model_name": model_name,
        "embedding_file": output_path.name,
        "chunk_file": chunk_file.name,
        "chunk_file_sha256": sha256_file(chunk_file),
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "first_chunk_id": chunks[0]["chunk_id"],
        "last_chunk_id": chunks[-1]["chunk_id"],
        "normalized": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode one query in the same normalized vector space as the chunks."""
    vector = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vector, dtype=np.float32)


def retrieve_top_k(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return exact cosine-ranked chunks using normalized-vector dot products."""
    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding rows and chunk metadata are misaligned")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query_vector = embed_query(model, query)
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[int(index)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "pdf_page": chunk["pdf_page"],
                "text": chunk["text"],
            }
        )
    return results


def result_matches_expected(
    result: dict[str, Any], expected_sources: list[dict[str, Any]]
) -> bool:
    return any(
        result["document"] == source["document"]
        and result["pdf_page"] in source["pdf_pages"]
        for source in expected_sources
    )


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    search_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate page-level evidence retrieval using Hit@k and MRR."""
    rows = []
    reciprocal_ranks = []

    for item in questions:
        results = retrieve_top_k(
            item["question"], model, embeddings, chunks, top_k=search_k
        )
        matching_ranks = [
            result["rank"]
            for result in results
            if result_matches_expected(result, item["expected_sources"])
        ]
        first_rank = min(matching_ranks) if matching_ranks else None
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "first_relevant_rank": first_rank,
                "hit_at_1": bool(first_rank and first_rank <= 1),
                "hit_at_3": bool(first_rank and first_rank <= 3),
                "hit_at_5": bool(first_rank and first_rank <= 5),
                "top_result_document": results[0]["document"],
                "top_result_page": results[0]["pdf_page"],
                "top_result_score": results[0]["score"],
            }
        )

    count = len(rows)
    summary = {
        "questions": count,
        "hit_at_1": sum(row["hit_at_1"] for row in rows) / count,
        "hit_at_3": sum(row["hit_at_3"] for row in rows) / count,
        "hit_at_5": sum(row["hit_at_5"] for row in rows) / count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / count,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=project_root
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.project_root
    model = load_embedding_model(
        root / "models" / "all-MiniLM-L6-v2", device=args.device
    )

    for target in (128, 220):
        chunk_path = root / "data" / "processed" / f"chunks_{target}.jsonl"
        chunk_records = read_jsonl(chunk_path)
        matrix = embed_chunks(model, chunk_records)
        report = validate_embeddings(matrix, chunk_records)
        output = root / "data" / "embeddings" / f"embeddings_{target}.npy"
        manifest = save_embedding_artifact(
            matrix, output, chunk_path, chunk_records
        )
        print(f"{target}-token embeddings: {report}")
        print(f"Manifest: {manifest}")
