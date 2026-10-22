# Validation notes

**Historical validation date: 2026-10-22.** Consolidated on 2026-10-23 from the original audit. The findings below describe checks performed on the existing self-directed learning prototype; consolidating this document did not repeat model inference or the full answer experiment.

## Evidence and configurations

- The [extraction report](../data/processed/extraction_report.json) records three PDFs, 163 original pages and 159 retained pages.
- The saved chunk corpora contain 972 records at the 128-token target / 24-token overlap target and 544 at 220 / 40. Overlap targets are not guaranteed exact overlap counts.
- Saved embeddings are normalized, 384-dimensional float32 matrices. Their [128-token manifest](../data/embeddings/embeddings_128.manifest.json) and [220-token manifest](../data/embeddings/embeddings_220.manifest.json) retain chunk-file hashes and row metadata.
- The [historical experiment](../evaluation/rag_experiment_results.json) records **0.45**, four configurations and 13 questions per configuration: 10 supported and 3 unsupported. Its 52 rows include gate-rejected cases, not 52 generator calls. Basic prompts bypass the gate.
- The [manual answer review](../evaluation/manual_grounded_top3_review.json) analyzes historical grounded Top-3 answers from that 0.45 experiment. It is not a fresh 0.58 answer-quality evaluation.
- Current `run_rag()` defaults are grounded Top-3, a **0.58** minimum top similarity and a 1,024-token input budget. The [saved notebook 06 walkthrough](../notebooks/06_end_to_end_rag_query_walkthrough.ipynb) explicitly checks the 0.58 gate before its recorded generation.
- [Notebook 07](../notebooks/07_final_error_analysis.ipynb) separately records a later retrieval-only check on five new unsupported questions at 0.58. It does not regenerate the supported answer set.
- Displayed sources are deduplicated document/pages supplied to the generator. They are provenance, not verified claim-level citations. Cosine similarity is not a probability or answer-confidence score.

## Checks actually performed on 2026-10-22

| Check | Observed finding | Scope and limitation |
|---|---|---|
| Python and notebook structure | Modules/scripts parsed as Python; all seven notebooks parsed as JSON with no saved error outputs. | This did not execute every notebook cell. |
| PDF extraction replay | Re-extracted all three PDFs in memory; fresh page records exactly matched the saved corpus, without rewriting it. | Agreement with the saved text does not prove perfect PDF interpretation or table extraction. |
| Page records | Checked unique document/page identities and cleaned character counts. | Rule-based cleaning still retains known PDF spacing artifacts. |
| Chunk corpora | Checked unique IDs, source-page coverage, nonempty text and recorded token budgets. | Token lengths were checked from stored metadata, not recalculated for every chunk during this audit. |
| Embedding artifacts | Checked shapes, float32 dtype, finite values, unit norms, chunk-file hashes and first/last row identifiers. | Row-count/hash checks are artifact consistency checks, not an independent recomputation of every vector. |
| Retrieval replay | Ten labelled questions per design reproduced Hit@1/3/5 and MRR shown below. | Small hand-labelled set; retrieval quality is separate from answer quality. |
| Unsupported holdout replay | Five of five questions had top similarity below 0.58. Maximum was **0.5702694**, leaving a minimum margin **0.0097306**. | Gate-only check; five questions and a narrow margin do not establish general reliability or safety. |
| Supported Qwen smoke query | Q02's maintenance-scheduling-units question produced time/distance/cycles, with three source pages and **526 input tokens**. | One current smoke query, not a new complete evaluation. |
| Unsupported Qwen-path smoke query | A warranty question returned `INSUFFICIENT EVIDENCE`, with zero generator input tokens and no generation call. | One rejection example, not universal unsupported-question detection. |
| Experiment output protection | The experiment CLI refused the existing historical output destination as intended. | No historical evaluation JSON was overwritten. |

Retrieval replay, using the [ten retrieval questions](../evaluation/retrieval_questions.json):

| Chunk target / overlap target | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| 128 / 24 | 0.50 | 0.90 | 0.90 | 0.63333 |
| 220 / 40 | 0.50 | 0.70 | 0.80 | 0.61944 |

The original saved comparison is in [notebook 03](../notebooks/03_embeddings_and_semantic_retrieval.ipynb). The extra decimal places in the holdout replay above come from the recorded local audit; notebook 07 displays rounded scores and a margin of approximately 0.010.

## Environment and reproduction findings

Python 3.10 was locally exercised. [requirements-observed.txt](../requirements-observed.txt) records selected installed versions, including a local CUDA-enabled PyTorch installation; it is not a complete lockfile or proof of a clean installation. [requirements.txt](../requirements.txt) declares direct dependencies using version ranges.

`pip check` on 2026-10-22 found conflicts in the existing shared environment:

- Installed NumPy was **1.23.5**.
- numba, pyLDAvis and scikit-image required newer NumPy versions.
- thinc required NumPy 2.x.

These packages are outside the project's direct dependency list. The exercised RAG checks passed, but fresh dependency resolution remains unvalidated. Use an isolated environment and run `python -m pip check` after installation.

Transformers emitted a warning that its default cache directory was not writable. Local offline model loading and inference nevertheless completed from the existing model directories. A writable project-local `HF_HOME` may be needed on a similarly restricted machine. No dependency installation or new model download was performed during the historical audit.

## Scope of the historical privacy check

The text scan covered Python, notebooks, evaluation JSON and embedding manifests, checking for user-home paths, email shapes and common Hugging Face, GitHub, OpenAI, AWS and private-key credential patterns. Home paths were found in historical notebook outputs/manifests; public notebook copies use a redacted home prefix. No email or credential-pattern matches occurred in the scanned files.

This was a scoped heuristic scan, not proof that all possible secrets are absent or a comprehensive security audit. The public NIST PDFs may contain published author contact information. Company/location names in unsupported evaluation questions are synthetic negative-test scenarios, not company operational data.

## Remaining limitations

Clean-environment installation, network execution of the [model download script](../scripts/download_models.py), exact historical model revisions and full notebook re-execution remain unvalidated. Model revisions were not pinned, so downloading current defaults cannot promise bit-for-bit historical reproduction. No full 52-row answer experiment was rerun at 0.58.

Q04 remains a retrieval failure followed by an inadequately grounded answer. Q06 illustrates both a phrase-matcher false negative and a real omission. The small generator can omit or fabricate information even with a relevant page in context. Small question sets, calibration-set reuse, PDF text artifacts and the narrow holdout margin limit generalization. No claim-level faithfulness checker or production-readiness evidence is provided.

See the [README](../README.md) for setup, interpretation and results, the [learning checkpoints](../LEARNING_CHECKPOINTS.md) for the function-level study guide, and the [validator](../scripts/validate_project.py) for the reusable structural and optional model smoke checks. Saved artifacts and earlier observations do not replace validation of a new environment.
