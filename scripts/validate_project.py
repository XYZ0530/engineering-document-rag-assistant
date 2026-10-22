"""Read-only structural checks; --models adds local inference smoke checks."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()
    import numpy as np
    for p in list((root / 'src').glob('*.py')) + list((root / 'scripts').glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8-sig'))
    pages = [json.loads(s) for s in (root / 'data/processed/pages.jsonl').read_text(encoding='utf-8').splitlines()]
    report = json.loads((root / 'data/processed/extraction_report.json').read_text(encoding='utf-8'))
    assert len(pages) == 159 and sum(d['total_pdf_pages'] for d in report['documents']) == 163
    keys = {(p['document'], p['pdf_page']) for p in pages}
    assert len(keys) == len(pages)
    assert all(len(p['cleaned_text']) == p['cleaned_character_count'] >= 30 for p in pages)
    for size, count in [(128, 972), (220, 544)]:
        cp = root / f'data/processed/chunks_{size}.jsonl'
        chunks = [json.loads(s) for s in cp.read_text(encoding='utf-8').splitlines()]
        matrix = np.load(root / f'data/embeddings/embeddings_{size}.npy', allow_pickle=False)
        manifest = json.loads((root / f'data/embeddings/embeddings_{size}.manifest.json').read_text(encoding='utf-8'))
        assert len(chunks) == len({c['chunk_id'] for c in chunks}) == count
        assert {(c['document'], c['pdf_page']) for c in chunks} == keys
        assert all(c['text'].strip() and c['token_count'] <= size for c in chunks)
        assert matrix.shape == (count, 384) and matrix.dtype == np.float32
        assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
        assert hashlib.sha256(cp.read_bytes()).hexdigest() == manifest['chunk_file_sha256']
        assert manifest['rows'] == count and manifest['dimensions'] == 384
        assert manifest['first_chunk_id'] == chunks[0]['chunk_id'] and manifest['last_chunk_id'] == chunks[-1]['chunk_id']
        print(f'PASS {size}: {count} aligned, normalized vectors; source hash verified')
    for p in (root / 'notebooks').glob('*.ipynb'):
        book = json.loads(p.read_text(encoding='utf-8'))
        errors = [o for c in book['cells'] for o in c.get('outputs', []) if o['output_type'] == 'error']
        assert not errors, p.name
        print(f'PASS notebook JSON/no saved errors: {p.name}')
    experiment = json.loads((root / 'evaluation/rag_experiment_results.json').read_text(encoding='utf-8'))
    assert experiment['minimum_evidence_score'] == 0.45 and len(experiment['rows']) == 52
    print('PASS historical experiment: 52 rows at 0.45; no artifacts rewritten')
    if args.models:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['USE_TF'] = '0'
        from src.embed_retrieve import load_embedding_model, read_jsonl, evaluate_retrieval, retrieve_top_k
        from src.rag_generate import load_generator, run_rag
        model = load_embedding_model(root / 'models/all-MiniLM-L6-v2')
        questions = json.loads((root / 'evaluation/retrieval_questions.json').read_text(encoding='utf-8'))
        for size in (128, 220):
            c = read_jsonl(root / f'data/processed/chunks_{size}.jsonl')
            e = np.load(root / f'data/embeddings/embeddings_{size}.npy')
            _, summary = evaluate_retrieval(questions, model, e, c)
            print('FRESH retrieval', size, summary)
        c = read_jsonl(root / 'data/processed/chunks_128.jsonl')
        e = np.load(root / 'data/embeddings/embeddings_128.npy')
        holdout = json.loads((root / 'evaluation/holdout_unsupported_questions.json').read_text(encoding='utf-8'))
        scores = [retrieve_top_k(q['question'], model, e, c, top_k=3)[0]['score'] for q in holdout]
        print('FRESH holdout at 0.58:', sum(s < 0.58 for s in scores), '/', len(scores), 'max score', max(scores))
        tokenizer, generator = load_generator(root / 'models/qwen2.5-0.5b-instruct')
        for q in [questions[1]['question'], holdout[0]['question']]:
            result = run_rag(q, model, e, c, tokenizer, generator)
            print('FRESH RAG', json.dumps({k: result[k] for k in ['question','answer','evidence_gate_triggered','sources_in_prompt','input_tokens']}, ensure_ascii=True))
            assert result['input_tokens'] <= 1024
            assert len(result['retrieved_sources']) <= result['sources_in_prompt']


if __name__ == '__main__':
    main()
