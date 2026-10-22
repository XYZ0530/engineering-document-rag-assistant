"""Build grounded prompts and generate answers from retrieved NIST evidence."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.embed_retrieve import retrieve_top_k


GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
INSUFFICIENT_RESPONSE = "INSUFFICIENT EVIDENCE"


def load_generator(local_model_dir: Path, device: str = "cpu"):
    """Load the local instruction-tuned text-to-text generator."""
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir, local_files_only=True
    ).to(device)
    model.eval()
    return tokenizer, model


def format_source(result: dict[str, Any], source_number: int) -> str:
    """Give each retrieved chunk a stable label that the LLM can cite."""
    return (
        f"[S{source_number} | {result['document']}, PDF p.{result['pdf_page']}]\n"
        f"{result['text']}"
    )


def encode_prompt(prompt: str, tokenizer) -> list[int]:
    """Apply an instruction model's chat template before counting/generation."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer.encode(prompt, add_special_tokens=True, truncation=False)


def build_prompt(
    question: str,
    retrieved: list[dict[str, Any]],
    generator_tokenizer,
    style: str = "grounded",
    max_input_tokens: int = 512,
) -> tuple[str, list[dict[str, Any]], int]:
    """Add ranked sources while keeping the complete prompt inside the LLM limit."""
    if style not in {"basic", "grounded"}:
        raise ValueError("style must be 'basic' or 'grounded'")

    if style == "basic":
        instruction = "Answer the question using the context."
    else:
        instruction = (
            "Answer using only the evidence below in no more than three concise sentences. "
            "State the answer directly, do not discuss these instructions, and do not use "
            "outside knowledge. Source citations are attached separately by the system."
        )

    selected: list[dict[str, Any]] = []
    source_blocks: list[str] = []

    def assemble(blocks: list[str]) -> str:
        context = "\n\n".join(blocks)
        return f"{instruction}\n\nEVIDENCE:\n{context}\n\nQUESTION: {question}\nANSWER:"

    for result in retrieved:
        candidate_blocks = source_blocks + [format_source(result, len(source_blocks) + 1)]
        candidate_prompt = assemble(candidate_blocks)
        token_count = len(encode_prompt(candidate_prompt, generator_tokenizer))
        if token_count <= max_input_tokens:
            source_blocks = candidate_blocks
            selected.append(result)
        else:
            break

    if not selected:
        raise ValueError("No complete source fits inside the generator input budget")

    prompt = assemble(source_blocks)
    input_tokens = len(encode_prompt(prompt, generator_tokenizer))
    return prompt, selected, input_tokens


def generate_answer(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 96,
) -> tuple[str, float]:
    """Generate deterministically so prompt experiments are reproducible."""
    input_ids = torch.tensor([encode_prompt(prompt, tokenizer)], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    latency = time.perf_counter() - start
    generated_ids = output_ids[0, input_ids.shape[1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer, latency


def extract_citations(answer: str) -> list[str]:
    """Extract source-ID or full NIST document/page citations."""
    patterns = (
        r"\[S\d+\]",
        r"\[NIST\.AMS\.100-\d+,\s*PDF\s+p(?:age)?\.?\s*\d+\]",
        r"\[Source:\s*NIST\.AMS\.100-\d+,\s*PDF\s+page\s+\d+\]",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, answer, flags=re.IGNORECASE))
    return list(dict.fromkeys(matches))


def unique_retrieved_sources(
    selected_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate document/page metadata while preserving retrieval order."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in selected_sources:
        key = (source["document"], source["pdf_page"])
        if key not in seen:
            unique.append(
                {
                    "document": source["document"],
                    "pdf_page": source["pdf_page"],
                    "retrieval_rank": source["rank"],
                    "retrieval_similarity": source["score"],
                }
            )
            seen.add(key)
    return unique


def format_retrieved_source_list(sources: list[dict[str, Any]]) -> str:
    """Render deterministic provenance separately from generated answer text."""
    lines = ["Retrieved sources:"]
    lines.extend(
        f"{index}. {source['document']} — PDF page {source['pdf_page']}"
        for index, source in enumerate(sources, start=1)
    )
    return "\n".join(lines)


def run_rag(
    question: str,
    embedding_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    generator_tokenizer,
    generator_model,
    top_k: int = 3,
    prompt_style: str = "grounded",
    max_input_tokens: int = 1024,
    minimum_evidence_score: float = 0.58,
) -> dict[str, Any]:
    """Execute retrieval, context budgeting, prompting, and generation."""
    retrieved = retrieve_top_k(
        question, embedding_model, chunk_embeddings, chunks, top_k=top_k
    )
    # Similarity is not a probability. This MVP threshold was calibrated on the
    # current supported/unsupported set and requires validation on new questions.
    if prompt_style == "grounded" and retrieved[0]["score"] < minimum_evidence_score:
        return {
            "question": question,
            "prompt_style": prompt_style,
            "requested_top_k": top_k,
            "sources_in_prompt": 0,
            "input_tokens": 0,
            "top_similarity": retrieved[0]["score"],
            "answer": INSUFFICIENT_RESPONSE,
            "raw_model_answer": "",
            "citations": [],
            "citation_present": False,
            "citation_method": "none; evidence gate rejected retrieval",
            "insufficient_evidence": True,
            "latency_seconds": 0.0,
            "selected_sources": [],
            "retrieved_sources": [],
            "prompt": "",
            "evidence_gate_triggered": True,
        }
    prompt, selected, input_tokens = build_prompt(
        question,
        retrieved,
        generator_tokenizer,
        style=prompt_style,
        max_input_tokens=max_input_tokens,
    )
    answer, latency = generate_answer(prompt, generator_tokenizer, generator_model)
    retrieved_sources = unique_retrieved_sources(selected)
    final_answer = f"{answer}\n\n{format_retrieved_source_list(retrieved_sources)}"
    return {
        "question": question,
        "prompt_style": prompt_style,
        "requested_top_k": top_k,
        "sources_in_prompt": len(selected),
        "input_tokens": input_tokens,
        "top_similarity": retrieved[0]["score"],
        "answer": final_answer,
        "raw_model_answer": answer,
        "citations": retrieved_sources,
        "citation_present": bool(retrieved_sources),
        "citation_method": "all deduplicated sources included in the generator prompt",
        "insufficient_evidence": answer.upper().strip() == INSUFFICIENT_RESPONSE,
        "latency_seconds": latency,
        "selected_sources": selected,
        "retrieved_sources": retrieved_sources,
        "prompt": prompt,
        "evidence_gate_triggered": False,
    }
