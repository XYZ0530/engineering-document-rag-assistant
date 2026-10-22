# GitHub upload steps (review first)

No Git configuration, commit, remote repository, push, or publication was performed during preparation.

## Prepared copy

The local export is at `../github-upload/engineering-document-rag-assistant-final/` relative to the original project. It contains only allowlisted project files: documentation, Python modules/scripts, seven notebooks with useful saved outputs, three PDFs, processed corpora, small embedding arrays/manifests, and original evaluation JSON. No parent-directory files, Git history, models, caches, environment, or bytecode are copied. `UPLOAD_MANIFEST.json` lists file SHA-256 values.

Original notebooks/manifests retain their historical contents. Exported notebooks redact the local user-home path prefix; exported manifests use project-relative chunk paths. Original evaluation JSON is copied byte-for-byte. No project code license was chosen on your behalf; decide whether to add one before inviting reuse. NIST sources and model links have their own terms; no model weights are redistributed.

## Rebuild the copy if you edit the project

From the original project root, choose a **new**, nonexistent destination:

```powershell
python scripts/prepare_upload.py --destination ../github-upload/engineering-document-rag-assistant-review2
```

The exporter refuses an existing destination. Review it in a file browser and read `AUDIT.md`. To validate the exported data using the current Python environment:

```powershell
python ../github-upload/engineering-document-rag-assistant-final/scripts/validate_project.py
```

## When you explicitly decide to publish

These are instructions only; they have **not** been executed. Start the following commands from the original project root. First copy the reviewed export outside the parent `Data Analytics` repository, then initialize a standalone repository there. A project-local `.gitignore` does not protect you if you stage from the wrong parent directory.

```powershell
$reviewCopy = (Resolve-Path '../github-upload/engineering-document-rag-assistant-final').Path
$repoPath = Join-Path $env:USERPROFILE 'source\engineering-document-rag-assistant'
if (Test-Path -LiteralPath $repoPath) { throw 'Choose a new destination; do not merge into an existing folder.' }
New-Item -ItemType Directory -Force -Path (Split-Path $repoPath) | Out-Null
Copy-Item -LiteralPath $reviewCopy -Destination $repoPath -Recurse
Set-Location -LiteralPath $repoPath
git init -b main
git rev-parse --show-toplevel
```

Verify the printed root is exactly `$repoPath`. If it is different, stop before staging. Then inspect:

```powershell
git add --dry-run .
git status --short
```

Review every proposed path, especially notebooks and evaluation prompts. Confirm there are no models, caches, personal files, or unrelated directory entries. When satisfied:

```powershell
git add .
git diff --cached --stat
git diff --cached --check
git status --short
git commit -m "Add self-directed manufacturing document RAG learning prototype"
```

Create an **empty** repository in GitHub's web interface with the visibility you choose, without generated README/license/gitignore. Copy its HTTPS URL and replace the placeholder below:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/engineering-document-rag-assistant.git
git remote -v
git push -u origin main
```

These final commands publish the reviewed files. Do not run them until you intend to publish. Keep downloaded models local; run the documented model download script after cloning on another machine.
