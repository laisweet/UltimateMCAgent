"""离线格式/动作一致率评估。可用本机 Ollama OpenAI 兼容口。

例：
  python model/evaluate_offline.py \\
    --data model/sft_data/valid.jsonl \\
    --base-url http://127.0.0.1:11434/v1 \\
    --model qwen2.5-mc:7b --api-key ollama
"""
import argparse, json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def decode_decision(text):
    text = text.strip()
    fence = '```'
    if text.startswith(fence):
        parts = text.splitlines()
        if parts and parts[0].startswith(fence):
            parts = parts[1:]
        if parts and parts[-1].strip() == fence:
            parts = parts[:-1]
        text = '\n'.join(parts)
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        first = text.find('{')
        if first < 0:
            raise ValueError('model did not return JSON')
        doc, _ = dec.raw_decode(text[first:])
    if not isinstance(doc, dict):
        raise ValueError('model response must be an object')
    action, args = doc.get('action'), doc.get('args', {})
    if not isinstance(action, str):
        raise ValueError('missing action')
    if not isinstance(args, dict):
        raise ValueError('args must be object')
    return {'reason': str(doc.get('reason', ''))[:400], 'action': action, 'args': args}


def evaluate(rows, predict):
    total = valid = action_match = exact = 0
    outputs = []
    for row in rows:
        messages = row['messages']
        target = json.loads(messages[-1]['content'])
        try:
            raw = predict(messages[:-1])
            result = decode_decision(raw)
            ok = True
        except Exception as exc:
            result = {'error': f'{type(exc).__name__}: {exc}'}
            ok = False
        total += 1
        valid += int(ok)
        action_match += int(ok and result.get('action') == target['action'])
        exact += int(ok and result.get('action') == target['action']
                     and result.get('args') == target['args'])
        outputs.append({'target': target, 'predicted': result})
    return {
        'samples': total,
        'json_action_valid': valid / total if total else 0,
        'action_name_accuracy': action_match / total if total else 0,
        'exact_args_accuracy': exact / total if total else 0,
    }, outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='sft_data/valid.jsonl')
    p.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    p.add_argument('--model', default='qwen2.5-mc:7b')
    p.add_argument('--api-key', default='ollama')
    p.add_argument('--output', default='eval_predictions.json')
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--timeout', type=int, default=90)
    a = p.parse_args()
    rows = [json.loads(x) for x in Path(a.data).read_text(encoding='utf-8').splitlines() if x.strip()][:a.limit]
    if not rows:
        raise SystemExit('No validation samples. Collect at least two independent episodes.')
    endpoint = a.base_url.rstrip('/')
    if not endpoint.endswith('/chat/completions'):
        endpoint += '/chat/completions'

    def predict(msgs):
        request = Request(
            endpoint, method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + a.api_key},
            data=json.dumps({
                'model': a.model, 'messages': msgs,
                'temperature': 0, 'max_tokens': 300,
            }, ensure_ascii=False).encode(),
        )
        try:
            with urlopen(request, timeout=a.timeout) as response:
                doc = json.loads(response.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}') from exc
        return doc['choices'][0]['message']['content']

    metrics, outputs = evaluate(rows, predict)
    Path(a.output).write_text(json.dumps({'metrics': metrics, 'predictions': outputs},
                                         ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
