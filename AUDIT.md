# Local audit and review record

Audit date: 2026-10-22. Scope: existing self-directed learning prototype, inspected from actual files, with inexpensive offline validation. No full answer experiment was rerun.

## Inventory and boundaries

Five Python implementation modules, seven notebooks with saved outputs, five evaluation JSON files, three PDFs, three processed JSONL corpora plus extraction diagnostics, and two embedding matrices/manifests were present. No README existed initially.

Initial folder contents total approximately 1.81 GB (decimal). Models: 1,711,380,657 bytes; `.hf_cache`: 92,276,794 bytes; data: 10,353,608 bytes; notebooks: 247,617 bytes; evaluation: 135,892 bytes. Old FLAN-T5 weights and bytecode exist locally. Nothing was deleted: these are excluded from the upload copy.

`git rev-parse --show-toplevel` resolved to the parent `Data Analytics` directory, not this project. Parent status included unrelated resume files and output directories. No parent files, Git configuration, history, index, branches, or remotes were changed. Preparation creates an export without `.git`; it must be copied outside that parent repository before Git initialization (see GITHUB_UPLOAD.md).

## What the evidence establishes

- Extraction diagnostics: 163 original pages, 159 retained.
- Saved corpora: 972 chunks (128/24) and 544 (220/40).
- Vectors: normalized 384-dimensional float32 arrays with chunk hash manifests.
- Historical experiment: 52 answer rows, four configurations, 13 questions, threshold explicitly 0.45. Basic prompt configurations bypass the gate.
- Manual answer review concerns historical grounded Top-3 outputs, not a fresh 0.58 answer experiment.
- Notebook 07's later holdout is a retrieval gate check at 0.58. It does not regenerate the supported answer set.
- Current run_rag defaults to grounded Top-3, 1,024 input tokens and 0.58. Source display lists deduplicated pages actually included in the prompt, not nearest-source attribution or claim verification.

## Changes made

- Added README with architecture, choices, setup, downloads, execution order, query example, historical results, limitations and future work.
- Added observed-version reference separate from broad dependency ranges; declared direct NumPy and Hugging Face Hub dependencies and raised Transformers minimum to the locally exercised 4.46 family.
- Added explicit public model download, read-only validation, and allowlisted export scripts.
- Hardened experiment CLI: explicit new output destination required, rejects existing results before inference, configurable threshold. Historical function default stays 0.45.
- New embedding manifests store a portable basename; originals remain untouched. Exported historical manifests use project-relative chunk paths while retaining hashes.
- Reconciled stale attribution descriptions, historical/current gate distinctions and duplicate incomplete checklist entries. Original teaching explanations and notebook outputs remain.
- Broadened `.gitignore` to cover every model folder, caches, environments, bytecode and local environment secrets. Small embedding arrays are now intentionally included.

## Checks actually run

1. Python syntax parsing for modules/scripts; all seven notebooks parsed as JSON with no saved error outputs. This is not execution of every notebook cell.
2. Re-extracted all three PDFs in memory: fresh page records exactly matched the saved corpus, without writing files. All page records checked for uniqueness and cleaned character counts. Both chunk corpora checked for unique IDs, source-page coverage, nonempty text and recorded token budgets. Both vector matrices checked for shape, dtype, finiteness, normalization and manifest source hashes/order endpoints. Token lengths were checked from stored metadata, not recalculated for every chunk.
3. Offline retrieval rerun on ten labelled questions per corpus reproduced Hit@1/3/5 and MRR: 128 = 0.50/0.90/0.90/0.63333; 220 = 0.50/0.70/0.80/0.61944.
4. Offline five-question holdout check at 0.58 rejected 5/5; maximum similarity 0.5702694, minimum margin 0.0097306.
5. Offline Qwen smoke: Q02 units question generated time/distance/cycles with three source pages and 526 input tokens. Warranty negative query returned INSUFFICIENT EVIDENCE with zero generator input tokens. These are two current smoke queries, not a replacement evaluation set.
6. Experiment CLI against the existing historical destination exited with the intended overwrite refusal. No historical JSON was overwritten.
7. `pip check` found existing-machine conflicts: numba, pyLDAvis and scikit-image require newer NumPy than installed 1.23.5; thinc requires NumPy 2.x. They are outside this project's direct requirements. The exercised RAG checks passed, but use an isolated environment; fresh dependency resolution has not been validated.

Transformers emitted a cache-directory writability warning on this machine during offline model loading; loading and inference still completed from the local model directories. Set `HF_HOME` to a writable project-local cache location if needed. No dependency installation or new model download was performed.

## Privacy and evidence handling

Text scan of Python, notebooks, evaluation JSON and embedding manifests checked for user-home paths, email shapes and common Hugging Face/GitHub/OpenAI/AWS/private-key credential patterns. User-home paths occurred in notebooks and historical manifests. No email or credential-pattern matches occurred in those scanned files; this is a heuristic scan, not proof of absence of all secrets. PDF content consists of the public NIST reports; published author contact details may be present and were not treated as user secrets.

The exporter redacts local username prefixes only in notebook copies and normalizes copied manifests. It preserves original evaluation JSON byte-for-byte. Evaluation company/location names are intentionally unsupported test scenarios, not company operational records. Career-planning references in learning documents are generic capability notes; job-target-specific roadmap wording was generalized during cleanup. Inspect the export before publication.

## Remaining work

Clean-environment installation, model download script network execution, exact historical model revision recovery and full notebook re-execution remain unvalidated. No lockfile, deployment, UI, trained model, claim-level faithfulness checker, or production-readiness evidence is provided. Q04 retrieval failure, omissions, PDF text artifacts, small evaluation sets and narrow gate margin remain. The 52-row historical experiment and all original artifacts are preserved for inspection.

## Upload copy validation

Final review copy: `../github-upload/engineering-document-rag-assistant-final/` (about 10.85 MB; 41 project files plus upload manifest). Structural validator passed against the exported copy. Every evaluation JSON SHA-256 matches the original. Exported notebook JSON remains valid with saved outputs; no local user-home prefix remains. An earlier draft export is also present under github-upload; use the final copy named above. No original model, cache, notebook, corpus, or evaluation evidence was deleted.
