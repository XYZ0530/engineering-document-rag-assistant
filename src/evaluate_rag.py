"""Run and summarize reproducible RAG prompt/context experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.embed_retrieve import load_embedding_model, read_jsonl
from src.rag_generate import load_generator, run_rag


CONFIGURATIONS = (
    {"configuration": "basic_top3", "prompt_style": "basic", "top_k": 3},
    {"configuration": "basic_top5", "prompt_style": "basic", "top_k": 5},
    {"configuration": "grounded_top3", "prompt_style": "grounded", "top_k": 3},
    {"configuration": "grounded_top5", "prompt_style": "grounded", "top_k": 5},
)


def normalize_text(text: str) -> str:
    """Normalize generated text for transparent phrase-based scoring."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_answer_points(
    answer: str, expected_points: list[dict[str, Any]]
) -> tuple[float, list[str], list[str]]:
    """Score whether at least one accepted phrase appears for each answer point."""
    if not expected_points:
        return 0.0, [], []
    normalized = normalize_text(answer)
    hits, misses = [], []
    for point in expected_points:
        matched = any(term.lower() in normalized for term in point["match_any"])
        (hits if matched else misses).append(point["label"])
    return len(hits) / len(expected_points), hits, misses


def expected_source_present(
    retrieved_sources: list[dict[str, Any]], expected_sources: list[dict[str, Any]]
) -> bool:
    """Check whether at least one accepted document/page reached the generator prompt."""
    return any(
        retrieved["document"] == expected["document"]
        and retrieved["pdf_page"] in expected["pdf_pages"]
        for retrieved in retrieved_sources
        for expected in expected_sources
    )


def looks_like_refusal(answer: str) -> bool:
    """Detect explicit refusal wording for unsupported-question evaluation."""
    normalized = normalize_text(answer)
    refusal_phrases = (
        "insufficient evidence",
        "cannot answer",
        "can't answer",
        "cannot determine",
        "not enough information",
        "does not provide",
        "does not contain",
        "no specific mention",
    )
    return any(phrase in normalized for phrase in refusal_phrases)


def evaluate_result(
    question: dict[str, Any], configuration: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Reduce a full RAG result to metrics and reviewable evidence."""
    coverage, hits, misses = score_answer_points(
        result["raw_model_answer"], question["expected_answer_points"]
    )
    retrieved_sources = result.get("retrieved_sources", [])
    refusal = looks_like_refusal(result["answer"])
    return {
        "configuration": configuration["configuration"],
        "prompt_style": configuration["prompt_style"],
        "top_k": configuration["top_k"],
        "question_id": question["question_id"],
        "question": question["question"],
        "supported": question["supported"],
        "top_similarity": result["top_similarity"],
        "evidence_gate_triggered": result["evidence_gate_triggered"],
        "llm_called": not result["evidence_gate_triggered"],
        "answer": result["answer"],
        "raw_model_answer": result["raw_model_answer"],
        "answer_point_coverage": coverage,
        "answer_point_hits": hits,
        "answer_point_misses": misses,
        "all_answer_points_hit": bool(question["supported"] and not misses),
        "expected_source_present": (
            expected_source_present(retrieved_sources, question["expected_sources"])
            if question["supported"]
            else None
        ),
        "refusal_detected": refusal,
        "correct_unsupported_refusal": bool(not question["supported"] and refusal),
        "input_tokens": result["input_tokens"],
        "sources_in_prompt": result["sources_in_prompt"],
        "unique_source_pages": len(retrieved_sources),
        "latency_seconds": result["latency_seconds"],
        "retrieved_sources": retrieved_sources,
    }


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate supported quality, refusal behavior, cost, and latency by configuration."""
    summaries = []
    for name in dict.fromkeys(row["configuration"] for row in rows):
        group = [row for row in rows if row["configuration"] == name]
        supported = [row for row in group if row["supported"]]
        unsupported = [row for row in group if not row["supported"]]
        called = [row for row in group if row["llm_called"]]
        summaries.append(
            {
                "configuration": name,
                "supported_questions": len(supported),
                "mean_answer_point_coverage": float(
                    np.mean([row["answer_point_coverage"] for row in supported])
                ),
                "all_answer_points_rate": float(
                    np.mean([row["all_answer_points_hit"] for row in supported])
                ),
                "expected_source_rate": float(
                    np.mean([row["expected_source_present"] for row in supported])
                ),
                "unsupported_refusal_rate": float(
                    np.mean([row["correct_unsupported_refusal"] for row in unsupported])
                ),
                "evidence_gate_rate_unsupported": float(
                    np.mean([row["evidence_gate_triggered"] for row in unsupported])
                ),
                "llm_call_rate": float(np.mean([row["llm_called"] for row in group])),
                "mean_input_tokens_when_called": float(
                    np.mean([row["input_tokens"] for row in called])
                ),
                "mean_latency_when_called": float(
                    np.mean([row["latency_seconds"] for row in called])
                ),
                "total_generation_seconds": float(
                    sum(row["latency_seconds"] for row in group)
                ),
            }
        )
    return summaries


def run_experiment(
    project_root: Path,
    minimum_evidence_score: float = 0.45,
) -> dict[str, Any]:
    """Load artifacts once, execute four configurations, and return compact results."""
    questions = json.loads(
        (project_root / "evaluation/rag_evaluation_questions.json").read_text(
            encoding="utf-8"
        )
    )
    embedding_model = load_embedding_model(
        project_root / "models/all-MiniLM-L6-v2"
    )
    chunks = read_jsonl(project_root / "data/processed/chunks_128.jsonl")
    embeddings = np.load(project_root / "data/embeddings/embeddings_128.npy")
    generator_tokenizer, generator_model = load_generator(
        project_root / "models/qwen2.5-0.5b-instruct"
    )

    rows = []
    total = len(CONFIGURATIONS) * len(questions)
    completed = 0
    for configuration in CONFIGURATIONS:
        for question in questions:
            result = run_rag(
                question["question"],
                embedding_model,
                embeddings,
                chunks,
                generator_tokenizer,
                generator_model,
                top_k=configuration["top_k"],
                prompt_style=configuration["prompt_style"],
                minimum_evidence_score=minimum_evidence_score,
            )
            rows.append(evaluate_result(question, configuration, result))
            completed += 1
            print(
                f"[{completed:02d}/{total}] {configuration['configuration']} "
                f"{question['question_id']} latency={result['latency_seconds']:.2f}s",
                flush=True,
            )

    return {
        "minimum_evidence_score": minimum_evidence_score,
        "question_count": len(questions),
        "configurations": list(CONFIGURATIONS),
        "summary": summarize_results(rows),
        "rows": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-evidence-score", type=float, default=0.45,
                        help="Historical experiment default; current run_rag default is 0.58")
    parser.add_argument("--output", type=Path, required=True,
                        help="New results path; existing files are never overwritten")
    args = parser.parse_args()
    if not 0 <= args.minimum_evidence_score <= 1:
        parser.error("--minimum-evidence-score must be between 0 and 1")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing evidence: {args.output}")
    results = run_experiment(root, args.minimum_evidence_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
