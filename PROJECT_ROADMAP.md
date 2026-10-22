# Manufacturing Maintenance RAG Assistant - Project Roadmap


> Review update (2026-10-22): This is a self-directed learning prototype. See README.md and AUDIT.md for current setup, validation, and evidence provenance. Historical notebooks/results are preserved; their saved outputs are not new runs. The current grounded gate is 0.58; the four-configuration experiment and its manual answer review use historical 0.45 runs. Current source display lists all deduplicated pages supplied to the generator, not semantic attribution or claim verification.
Last updated: 2026-08-16

## Purpose of this document

Use this roadmap to answer three questions quickly:

1. What are the main stages of the project?
2. What has already been completed?
3. Where should work resume next?

For detailed technical decisions and completion criteria, see `PROJECT_PLAN.md`. For source-document information, see `SOURCES.md`.

## Current position

**Current stage:** Stage 9 - Portfolio and CV preparation

**Current checkpoint:** Final manual error analysis is complete. Grounded Top-3 achieved 90% expected-source coverage, 63.3% documented manual answer-point coverage, and rejected 5/5 fresh unsupported questions at the provisional 0.58 gate.

**Next deliverable:** Review README.md, AUDIT.md and the project-only export; validate a fresh environment. CV work remains optional.

## Project flow

```text
1. Define scope and collect sources                         COMPLETE
                 |
                 v
2. Extract, clean, and validate PDF text                    COMPLETE
                 |
                 v
3. Create token-aware chunks                               COMPLETE
                 |
                 v
4. Generate Hugging Face embeddings                        COMPLETE
                 |
                 v
5. Implement and evaluate semantic retrieval               COMPLETE
                 |
                 v
6. Add grounded answer generation                          COMPLETE
                 |
                 v
7. Compare prompts and context configurations              COMPLETE
                 |
                 v
8. Evaluate the complete RAG pipeline                      COMPLETE
                 |
                 v
9. README and upload preparation done; fresh setup/CV work pending
```

## Stage 1 - Scope and source corpus

**Status:** Complete

### Objective

Define a project that adds Hugging Face, transformer NLP, RAG, prompt engineering, and context optimization evidence as a self-directed learning project.

### Completed work

- Selected the project title and scope: Manufacturing Maintenance RAG Assistant.
- Chose official NIST manufacturing-maintenance publications as the initial knowledge corpus.
- Downloaded and verified three PDFs with 163 total pages.
- Confirmed the project is educational knowledge retrieval, not equipment diagnosis.
- Recorded publication titles, URLs, page counts, and corpus limitations.

### Main outputs

- `PROJECT_PLAN.md`
- `SOURCES.md`
- `data/raw/NIST.AMS.100-2.pdf`
- `data/raw/NIST.AMS.100-18.pdf`
- `data/raw/NIST.AMS.100-34.pdf`

### What you should understand

- A RAG system needs a knowledge corpus from which it can retrieve evidence.
- NIST reports support general maintenance questions but not machine-specific diagnosis.
- Source metadata is required for transparent citations.

## Stage 2 - PDF extraction and text cleaning

**Status:** Complete for MVP

### Objective

Convert visual PDF pages into clean, structured text while retaining exact document and PDF-page metadata.

### Completed work

- Tested direct extraction and determined that OCR is unnecessary.
- Implemented page-level extraction using `pypdf`.
- Preserved raw and cleaned text for every retained page.
- Detected and removed repeated headers and DOI running lines.
- Removed isolated PDF page numbers.
- Repaired standard cross-line hyphenation.
- Skipped and visually verified blank pages.
- Wrote 159 retained page records to JSONL.
- Created extraction diagnostics and automated assertions.
- Created and executed the teaching notebook with all outputs saved.

### Main outputs

- `src/extract_text.py`
- `data/processed/pages.jsonl`
- `data/processed/extraction_report.json`
- `notebooks/01_pdf_extraction_and_cleaning.ipynb`

### Accepted limitation

Some words contain spacing artifacts created by the PDF text layer:

```text
enoug h
lead ing
Reliability -centered
```

The readiness audit found these issues to be isolated and non-blocking. Broadly deleting spaces could merge legitimate words and damage the corpus, so they are retained for the MVP and monitored during retrieval evaluation.

### Readiness result

- 159 complete page records
- No missing fields or empty cleaned text
- No duplicate document-page identifiers
- No character-count mismatches
- No corrupted replacement characters
- No leftover repeated NIST running headers
- Two known split words confined to one page
- Spaced hyphens present on nine pages and treated as non-blocking

### What you should understand

- Why direct extraction is preferable to OCR for digital PDFs.
- Why both raw and cleaned text are retained.
- How repeated page furniture is detected.
- Why automated validation and manual inspection are both necessary.
- Why JSONL is suitable for page-level records.

## Stage 3 - Token-aware chunking

**Status:** Complete

### Objective

Split cleaned pages into retrieval units small enough to represent focused topics without losing their citations.

### Planned approach

- Use lightweight sentence-boundary rules and the exact Hugging Face tokenizer.
- Preserve natural text; do not stem or remove stop words.
- Create two experimental configurations that remain below the model's 256-token limit:
  - 128 tokens with 24-token overlap
  - 220 tokens with 40-token overlap
- Preserve document name, page number, chunk number, and chunk ID.
- Avoid combining content from different documents.

### Outputs

- `src/chunk_text.py`
- `notebooks/02_token_aware_chunking.ipynb`
- `data/processed/chunks_128.jsonl`
- `data/processed/chunks_220.jsonl`
- Chunking diagnostics and validation results

### What you should learn

- What a token, sentence, word count, chunk, and overlap mean.
- Why entire documents are poor retrieval units.
- How small and large chunks affect retrieval precision and context.
- Why stop-word removal and stemming are generally unsuitable before transformer embeddings.

## Stage 4 - Hugging Face embeddings

**Status:** Complete

### Objective

Convert each text chunk into a vector representing its semantic meaning.

### Implemented approach

- Use `sentence-transformers/all-MiniLM-L6-v2`.
- Generated normalized embeddings for both chunk-size configurations.
- Produced matrices of 972 x 384 and 544 x 384 float32 values.
- Saved each matrix with a manifest binding its rows to the hashed chunk file.
- Validated dimensions, finite values, unit norms, and row alignment.

### Outputs

- `src/embed_retrieve.py`
- `notebooks/03_embeddings_and_semantic_retrieval.ipynb`
- `data/embeddings/embeddings_128.npy` and manifest
- `data/embeddings/embeddings_220.npy` and manifest

### What you should learn

- What an embedding vector represents.
- Why semantically similar text has nearby vectors.
- How transformer embeddings differ from TF-IDF and Word2Vec.

## Stage 5 - Semantic retrieval

**Status:** Complete for MVP

### Objective

Given a user question, retrieve the most relevant chunks from the corpus.

### Implemented approach

- Embed the user question with the same model used for document chunks.
- Calculate cosine similarity as a dot product between normalized vectors.
- Use exact NumPy search because the larger corpus contains only 972 vectors.
- Compare 128-token and 220-token retrieval using 10 manually labelled questions.
- Return text, score, document, and PDF page.

### Outputs

- Reusable retrieval function
- Example questions and retrieved passages
- `evaluation/retrieval_questions.json`
- Retrieval metrics and failure analysis saved in the executed notebook

### Measured results

| Chunk design | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| 128 tokens / 24 target overlap | 50% | 90% | 90% | 0.633 |
| 220 tokens / 40 target overlap | 50% | 70% | 80% | 0.619 |

The 128-token design is selected for the first RAG version. This is a preliminary comparison on a small evaluation set, not a universal claim that smaller chunks are always better.

### What you should learn

- How cosine similarity ranks semantic matches.
- Why query and document embeddings must use the same model.
- What `top-k` means and how it affects relevance.

## Stage 6 - Grounded answer generation

**Status:** Complete for MVP

### Objective

Use an instruction-following model to answer questions from retrieved evidence rather than from unsupported model memory.

### Implemented approach

- Tested FLAN-T5-small and documented its weak instruction/citation compliance.
- Replaced only the generator with `Qwen2.5-0.5B-Instruct`; the embedding index remained unchanged.
- Constructed prompts from complete ranked chunks within an explicit generator-token budget.
- Initially evaluated a 0.45 grounded gate; current run_rag default is 0.58. Historical results retain their original configuration.
- Generate deterministically and attach source provenance from retrieved document/page metadata.
- Display all deduplicated document/pages actually supplied to the generator (current implementation).

### Outputs

- `src/rag_generate.py`
- `notebooks/04_rag_and_prompt_engineering.ipynb`
- Executed supported and insufficient-evidence examples with latency and token counts

### What you should learn

- The difference between retrieval and generation.
- How retrieved context augments an LLM prompt.
- Why groundedness is different from general fluency.
- Why prompt instructions alone do not guarantee citation or refusal compliance.
- Why source provenance should come from retained metadata rather than model memory.
- Why the evidence threshold must be calibrated rather than treated as a probability.

## Stage 7 - Prompt and context experiments

**Status:** Complete

### Objective

Measure how prompt design and retrieval configuration affect answer quality.

### Planned comparisons

- Basic prompt versus grounded structured prompt
- 128-token versus 220-token chunks
- Top 3 versus top 5 retrieved chunks

### Planned measurements

- Correct supporting passage retrieved
- Citation compliance
- Answer supported by context
- Unsupported-claim rate
- Correct insufficient-evidence behaviour
- Response latency

### Measured decision

- Selected grounded Top-3 with 58.3% mean phrase-based answer-point coverage.
- Top-5 did not improve expected-source coverage (both 90%) and reduced grounded answer-point coverage to 55.0%.
- Grounded Top-5 used 718.8 mean input tokens versus 476.5 for Top-3 when the LLM was called.
- The initial 0.45 evidence gate rejected two of three unsupported questions; the third scored 0.510 and caused fabricated equipment answers.
- A provisional threshold of 0.58 was derived between the maximum unsupported score (0.510) and minimum supported score (0.657).
- Source output was simplified to all deduplicated document/pages actually supplied to the generator.

### Outputs

- `src/evaluate_rag.py`
- `evaluation/rag_evaluation_questions.json`
- `evaluation/rag_experiment_results.json`
- `notebooks/05_rag_evaluation_and_context_experiments.ipynb`

### What you should learn

- Prompt engineering should be evaluated rather than described subjectively.
- More context is not always better because irrelevant chunks can distract the model.

## Stage 8 - Final evaluation

**Status:** Complete for MVP

### Objective

Produce defensible evidence that the system works and document where it fails.

### Planned evaluation set

- 10 questions answerable from the corpus
- 3 questions deliberately unsupported by the corpus

### Planned outputs

- Evaluation question file
- Experiment-results table
- Error analysis with successful and failed examples
- Final configuration recommendation

### Completed outputs and results

- `evaluation/manual_grounded_top3_review.json`
- `evaluation/holdout_unsupported_questions.json`
- `notebooks/07_final_error_analysis.ipynb`
- Automated phrase coverage: 58.3%
- Documented manual point coverage: 63.3%
- Expected evidence reached grounded Top-3: 90%
- Manual verdicts: 3 supported, 2 mostly supported, 4 partially supported, 1 unsupported
- Fresh unsupported gate rejection: 5/5 at threshold 0.58
- Narrowest holdout threshold margin: 0.010
- Primary unacceptable failure: Q04 retrieval failure

### What you should learn

- Why a manually verified evaluation set is needed.
- Why retrieval and generation should be evaluated separately.
- How limitations make a portfolio project more credible.

## Stage 9 - Portfolio and CV preparation

**Status:** Local documentation and upload preparation completed; fresh-environment validation and CV work remain

### Objective

Turn the working experiment into a reproducible portfolio project and accurate resume evidence.

### Planned outputs

- Complete `README.md`
- Architecture diagram
- Setup and execution instructions
- Example questions and cited answers
- Actual experiment results
- Limitations and future work
- Optional minimal Streamlit interface
- Final CV bullets using measured results only

## Deferred extensions

These are intentionally excluded until the core project works:

- MaintIE maintenance-log semantic search
- LangChain
- OpenAI API comparison
- LoRA or PEFT fine-tuning
- MCP integration
- SQL interaction logging
- Cloud deployment
- Production equipment diagnosis

## Resume alignment

### Capabilities demonstrated through transformer NLP

- Hugging Face transformer usage
- RAG and semantic retrieval
- Question answering and summarization
- Prompt engineering
- Context optimization
- Experiment documentation

### Capabilities demonstrated through engineering documents

- Unstructured engineering-document processing
- Knowledge retrieval for manufacturing decision support
- NLP automation and technical-text summarization
- Maintainable Python pipeline design

Existing resume projects already provide evidence for classical ML, regression, deep learning, large-dataset processing, and manufacturing exposure.

## Resume point for the next session

Resume work from:

1. Open `notebooks/07_final_error_analysis.ipynb` and review the final measured results.
2. Write the project README with architecture, setup, execution order, examples and limitations.
3. Add a compact architecture diagram and results table.
4. Document Q04 retrieval failure and the narrow 0.010 holdout threshold margin.
5. Add accurate CV bullets using measured values only.
6. Prepare the repository for GitHub without committing downloaded model weights or generated caches.
