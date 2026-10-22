"""Download public model snapshots into the paths used by the notebooks."""
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    for repo, folder, patterns in (
        ('sentence-transformers/all-MiniLM-L6-v2', 'all-MiniLM-L6-v2',
         ['*.json', '*.txt', '*.safetensors', '1_Pooling/*']),
        ('Qwen/Qwen2.5-0.5B-Instruct', 'qwen2.5-0.5b-instruct',
         ['*.json', '*.txt', '*.safetensors', '*.model']),
    ):
        path = snapshot_download(repo_id=repo, local_dir=root / 'models' / folder,
                                 allow_patterns=patterns)
        print(f'{repo}: {path}')
    # Reuse the embedding snapshot tokenizer; no third download is needed.
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(root / 'models/all-MiniLM-L6-v2',
                                  local_files_only=True).save_pretrained(
        root / 'models/all-MiniLM-L6-v2-tokenizer')


if __name__ == '__main__':
    main()
