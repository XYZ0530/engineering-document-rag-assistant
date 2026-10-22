# Learning Checkpoints


> Review update (2026-10-22): This is a self-directed learning prototype. See README.md and AUDIT.md for current setup, validation, and evidence provenance. Historical notebooks/results are preserved; their saved outputs are not new runs. The current grounded gate is 0.58; the four-configuration experiment and its manual answer review use historical 0.45 runs. Current source display lists all deduplicated pages supplied to the generator, not semantic attribution or claim verification.
This guide points to the exact implementation locations you should understand well enough to explain confidently in an interview. Use it beside the executed notebooks.

## PDF extraction and cleaning

Notebook: `notebooks/01_pdf_extraction_and_cleaning.ipynb`

Implementation: `src/extract_text.py`

Key locations:

- `find_repeated_edge_lines()` - detects repeated page headers and footers.
- `clean_page_text()` - performs the main text-cleaning operations.
- `extract_document()` - extracts page text and preserves source metadata.
- `extract_corpus()` - writes the page corpus and diagnostic report.

You should be able to explain why direct extraction was chosen over OCR, why raw and cleaned text are both retained, and why document/page metadata must be captured before chunking.

## Token-aware chunking

Notebook: `notebooks/02_token_aware_chunking.ipynb`

Implementation: `src/chunk_text.py`

Key locations:

- `MODEL_NAME` and `MODEL_MAX_TOKENS` - selected Hugging Face embedding model and enforced 256-token constraint.
- `CHUNK_CONFIGS` - the 128/24 and 220/40 experiments.
- `load_tokenizer()` and `count_tokens()` - measure length with the exact model tokenizer.
- `split_sentence_units()` - creates structural units without NLTK.
- `split_oversized_unit()` - handles one sentence or table larger than the budget.
- `choose_overlap_start()` - selects complete trailing units for overlap and applies a bounded fallback when the desired overlap is smaller than one sentence.
- `count_adjacent_token_overlap()` - measures actual suffix/prefix overlap rather than assuming the configured target was achieved.
- `chunk_page()` - the core greedy chunk-building algorithm.
- `validate_chunks()` - verifies invariants before embedding.

You should be able to explain why word counts were replaced by model-token counts, why 500-word chunks would be unsafe, why two chunk sizes are retained for evaluation, and why structural validation does not establish retrieval quality.

## Embeddings and semantic retrieval

Notebook: `notebooks/03_embeddings_and_semantic_retrieval.ipynb`

Implementation: `src/embed_retrieve.py`

Key locations:

- `load_embedding_model()` - loads the local Hugging Face SentenceTransformer on CPU.
- `embed_chunks()` - batch-encodes chunk text into normalized 384-dimensional vectors.
- `validate_embeddings()` - checks dimensions, finite values, unit norms, and row alignment.
- `save_embedding_artifact()` - saves the NumPy matrix and a manifest tied to the source chunk hash.
- `embed_query()` - encodes a question in the same vector space as the document chunks.
- `retrieve_top_k()` - calculates `embeddings @ query_vector` and returns ranked text with source pages.
- `evaluate_retrieval()` - measures Hit@1/3/5 and MRR independently of answer generation.

You should be able to explain why the same model embeds queries and documents, why normalization turns cosine similarity into a dot product, why vector/metadata alignment is a correctness requirement, why exact NumPy search is sufficient for 972 vectors, and why the 128-token design was selected from measured retrieval results.

## Grounded RAG and prompt engineering

Notebook: `notebooks/04_rag_and_prompt_engineering.ipynb`

Implementation: `src/rag_generate.py`

Key locations:

- `load_generator()` - loads the local Qwen instruction model independently from the embedding model.
- `encode_prompt()` - applies the model's chat template for accurate token counting and generation.
- `build_prompt()` - adds complete ranked sources within the generator input-token budget.
- `generate_answer()` - performs deterministic greedy generation for reproducible experiments.
- `unique_retrieved_sources()` - deduplicates every document/page actually supplied to the generator.
- `format_retrieved_source_list()` - appends deterministic multi-source provenance after generation.
- `run_rag()` - orchestrates retrieval, evidence gating, prompt construction, generation and provenance.

You should be able to explain why retrieval and generation use different models, why changing the generator does not require re-embedding, why requested Top-k can differ from sources that fit the prompt, why low-similarity evidence is rejected before generation, and why a retrieved-source list records available evidence rather than proving every claim is faithful.

## RAG evaluation and context experiments

Notebook: `notebooks/05_rag_evaluation_and_context_experiments.ipynb`

Implementation: `src/evaluate_rag.py`

Key locations:

- `CONFIGURATIONS` - fixes the basic/grounded and Top-3/Top-5 comparison matrix.
- `score_answer_points()` - applies transparent accepted-phrase checks to expected facts.
- `expected_source_present()` - checks whether labelled evidence reached the generator prompt.
- `looks_like_refusal()` - measures explicit unsupported-question refusal behavior.
- `evaluate_result()` - records quality, guardrail, cost and latency fields per run.
- `summarize_results()` - aggregates results without hiding individual failures.

You should be able to explain controlled experiments, why Top-5 is not automatically better, why cosine thresholds require calibration, how a grounded prompt can still hallucinate after weak evidence passes the gate, and why automated phrase scoring requires manual error analysis.

## Final error analysis

Notebook: `notebooks/07_final_error_analysis.ipynb`

Review artifacts:

- `evaluation/manual_grounded_top3_review.json`
- `evaluation/holdout_unsupported_questions.json`

Key ideas:

- Compare automated phrase coverage with documented manual point coverage.
- Classify failures by pipeline layer: retrieval, generation, context, or scoring.
- Treat missing evidence as a retrieval problem that prompt engineering cannot repair.
- Validate a calibrated gate on fresh questions rather than only the calibration set.
- Report both holdout rejection rate and threshold margin.

You should be able to explain why Q06 was partly a scoring false negative, why Q04 was a genuine retrieval failure, why 5/5 holdout rejection does not prove production safety, and why the 0.010 margin makes the evidence gate an MVP guardrail rather than a guarantee.

## Teaching format used from now on

Every major notebook stage will include:

1. An implementation map with exact files and function names.
2. Implementation checkpoints beside the relevant explanation and code.
3. Inputs and outputs for each core function.
4. The engineering reason behind each design choice.
5. Validation criteria and the distinction between correctness and quality.
6. An interview-ready explanation supported by actual results.
