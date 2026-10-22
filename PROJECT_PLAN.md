# Engineering Document RAG Assistant


> Review update (2026-10-22): This is a self-directed learning prototype. See README.md and AUDIT.md for current setup, validation, and evidence provenance. Historical notebooks/results are preserved; their saved outputs are not new runs. The current grounded gate is 0.58; the four-configuration experiment and its manual answer review use historical 0.45 runs. Current source display lists all deduplicated pages supplied to the generator, not semantic attribution or claim verification.
## Project objective

Build a small, reproducible retrieval-augmented generation (RAG) system that answers manufacturing-maintenance questions from public NIST documents. The system must retrieve relevant passages, generate a grounded answer, and cite the source document and page.

This document is a working reference and may be updated as implementation choices or time constraints change.

## CV capabilities demonstrated

- Python-based NLP and document-preprocessing pipeline
- Hugging Face sentence-transformer embeddings
- Semantic retrieval and RAG
- Technical-document question answering and summarization
- Prompt engineering and context optimization
- Experiment design, evaluation, and documentation

## Initial corpus

Use three official NIST publications:

1. `NIST.AMS.100-2.pdf` - Prognostics and Health Management for Smart Manufacturing Systems
2. `NIST.AMS.100-18.pdf` - The Costs and Benefits of Advanced Maintenance in Manufacturing
3. `NIST.AMS.100-34.pdf` - Economics of Manufacturing Machinery Maintenance: A Survey and Analysis of U.S. Costs and Benefits

The exact corpus can be reduced if extraction quality or time becomes a problem.

## Minimum viable pipeline

1. Download and record the source documents.
2. Extract page-level text while preserving document and page metadata.
3. Clean repeated headers, footers, whitespace, and extraction artifacts.
4. Split text into configurable chunks.
5. Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
6. Retrieve the top-k chunks for a user question using cosine similarity.
7. Pass retrieved context to a small instruction-following model.
8. Produce a grounded answer with document and page citations.
9. Return `Insufficient evidence` when the corpus does not support an answer.

## Required experiments

### Chunk-size comparison

- 128 tokens with 24-token overlap
- 220 tokens with 40-token overlap

The initial word-based plan was replaced after verifying that the selected
embedding model truncates inputs beyond 256 word-piece tokens.

### Retrieval comparison

- Top 3 chunks
- Top 5 chunks

### Prompt comparison

- Basic question-answering prompt
- Grounded prompt requiring citations and refusal when evidence is insufficient

## Evaluation set

Prepare:

- 10 questions whose answers are explicitly present in the corpus
- 3 questions that cannot be answered from the corpus

Record:

- Whether the expected passage appears in the top-k results
- Whether the generated answer is supported by retrieved context
- Whether a valid source and page are cited
- Whether the model correctly handles insufficient evidence
- Response time

## Completion criteria

The project is CV-ready when:

- A fresh user can follow the README and run it.
- Retrieval works on the evaluation questions.
- Generated answers include verifiable source-page citations.
- Evaluation results use actual measured values.
- Dataset sources and limitations are documented honestly.
- No confidential company data is used.

## Planned structure

```text
engineering-document-rag-assistant/
|-- data/
|   |-- raw/
|   `-- processed/
|-- evaluation/
|-- notebooks/
|   |-- 01_pdf_extraction_and_cleaning.ipynb
|   `-- 02_token_aware_chunking.ipynb
|-- src/
|-- PROJECT_PLAN.md
|-- README.md
`-- requirements.txt
```

## Weekly execution plan

- Tuesday: source documents, extraction, and cleaning
- Wednesday: chunking, embeddings, and semantic retrieval
- Thursday: answer generation and prompt engineering
- Friday: evaluation, README, examples, and GitHub preparation

## Progress

- [x] Download and verify the three NIST source PDFs.
- [x] Implement direct page-level text extraction with source metadata.
- [x] Detect and skip blank pages; retain an extraction diagnostics report.
- [x] Remove repeated NIST headers, DOI running lines, and isolated page numbers.
- [x] Add an executable teaching notebook for extraction and cleaning.
- [x] Saved notebook 01 includes representative page inspection and cleaning checks; known artifacts remain documented.
- [x] Build and validate 128-token and 220-token chunk configurations.
- [x] Preserve sentence units, overlap, and exact page citations during chunking.
- [x] Correct the initial word-based design to respect the embedding model's 256-token limit.
- [x] Generate normalized 384-dimensional Hugging Face embeddings for both chunk designs.
- [x] Save embedding matrices with manifests that preserve chunk-row alignment.
- [x] Implement exact cosine-similarity retrieval with document and page metadata.
- [x] Evaluate 10 labelled questions using Hit@1/3/5 and MRR.
- [x] Select the 128-token corpus for the first RAG version based on stronger Hit@3 and MRR.
- [x] Build grounded answer generation with source provenance and insufficient-evidence handling.
- [x] Add token-budgeted context construction and deterministic local generation.
- [x] Demonstrate that the generator can be replaced without rebuilding the embedding index.
- [x] Compare Top-3 versus Top-5 and basic versus grounded configurations.
- [x] Calibrate a provisional 0.58 evidence threshold using supported and unsupported questions.
- [x] Replace single-source semantic attribution with all deduplicated sources supplied to the generator.
- [x] Manually audit answer faithfulness and phrase-scoring false negatives.
- [x] Validate the provisional threshold on five fresh unsupported questions.
- [x] Write README and reproducibility instructions (fresh-environment validation remains pending).
- [ ] Prepare measured CV bullets and interview talking points.

## Out of scope for the first version

- Classical predictive-maintenance modeling
- Fine-tuning, LoRA, or PEFT
- Agents or MCP integration
- Cloud deployment
- Complex user authentication
- Production safety or maintenance recommendations

## Safety and positioning

This is an educational knowledge-retrieval prototype. It must not be presented as a production diagnostic system or as a replacement for approved engineering, maintenance, or safety procedures.
