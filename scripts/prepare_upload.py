"""Create a new project-only review export. Never copies caches or Git metadata."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='New directory outside this project')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.destination.resolve()
    if dest == root or root in dest.parents or dest.exists():
        parser.error('Destination must be a new directory outside the project')
    files = [p for p in root.iterdir() if p.is_file() and
             (p.suffix == '.md' or p.name in {'.gitignore', 'requirements.txt', 'requirements-observed.txt'})]
    for folder, suffixes in {'src': {'.py'}, 'scripts': {'.py'},
                             'notebooks': {'.ipynb'}, 'evaluation': {'.json'},
                             'data': {'.pdf', '.json', '.jsonl', '.npy'}}.items():
        files.extend(p for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in suffixes and
                     not any(x.startswith('.') or x == '__pycache__'
                             for x in p.relative_to(root / folder).parts))
    dest.mkdir(parents=True)
    manifest = {}
    for p in sorted(files):
        rel = p.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.name.endswith('.manifest.json'):
            value = json.loads(p.read_text(encoding='utf-8'))
            value['chunk_file'] = 'data/processed/' + Path(value['chunk_file']).name
            target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        elif p.suffix == '.ipynb':
            # Remove only the local username prefix, retaining teaching outputs.
            text = p.read_text(encoding='utf-8')
            home = Path.home()
            for prefix in (str(home).replace(chr(92), chr(92) * 2), home.as_posix()):
                text = text.replace(prefix, '<USER_HOME>')
            json.loads(text)
            target.write_text(text, encoding='utf-8')
        else:
            shutil.copy2(p, target)
        manifest[rel.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (dest / 'UPLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(files)} files; {sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()):,} bytes to {dest}')


if __name__ == '__main__':
    main()
