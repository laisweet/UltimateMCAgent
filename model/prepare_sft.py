"""Convert vetted Harness JSONL traces to supervised tool-action examples.

Split by complete episodes/files to avoid leaking adjacent states to validation.
Do not use raw failed attempts or private game chat as ground truth.
"""
import argparse
import glob
import json
import random
from pathlib import Path

SYSTEM = ("你是 Minecraft Agent 的行动决策器。仅基于提供的观察结果，"
          "每次输出一个 JSON 对象，字段为 reason、action、args。"
          "只可使用公开的赛事 API。")

SENSITIVE = ("password", "token", "secret", "apikey", "api_key", "authorization", "cookie")

def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()
                if not any(s in str(k).lower() for s in SENSITIVE)
                and k not in {"events", "recent"}}  # events may contain player chat or personal data
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, str):
        return x[:8000]
    return x

def load_episode(path, completed_only=True):
    rows = []
    done = False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip(): continue
        event = json.loads(raw)
        if event.get("type") == "complete":
            done = True
        if event.get("type") != "trace" or event.get("error") is not None:
            continue
        decision = event.get("decision", {})
        if not isinstance(decision, dict) or decision.get("action") in {None, "chat", "disconnect"}:
            continue
        obs = scrub(event.get("observation", {}))
        # Teacher decisions remain candidates for manual auditing, not unquestionable gold labels.
        target = {"reason": str(decision.get("reason", ""))[:400],
                  "action": decision["action"], "args": scrub(decision.get("args", {}))}
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(obs, ensure_ascii=False, separators=(",", ":"))},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}
            ]
        })
    if completed_only and not done:
        return []
    return rows

def build(paths, outdir, *, seed=42, val_fraction=0.2, completed_only=True):
    paths = list(dict.fromkeys(str(Path(p)) for p in paths))
    episodes = [load_episode(path, completed_only=completed_only) for path in paths]
    episodes = [e for e in episodes if e]
    if not episodes:
        raise ValueError("no eligible completed episodes; pass --include-incomplete for exploratory data")
    random.Random(seed).shuffle(episodes)
    valid_n = max(1, round(len(episodes) * val_fraction)) if len(episodes) >= 2 else 0
    valid_episodes = episodes[:valid_n]
    train_episodes = episodes[valid_n:]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, groups in (("train", train_episodes), ("valid", valid_episodes)):
        rows = [row for ep in groups for row in ep]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[name] = {"episodes": len(groups), "examples": len(rows)}
    (out / "dataset_manifest.json").write_text(
        json.dumps({"seed": seed, "completed_only": completed_only, **stats},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return stats

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, nargs="+", help="JSONL paths or glob patterns")
    parser.add_argument("--out", default="sft_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Exploratory only: unsuccessful trajectories may contain bad teacher actions")
    opts = parser.parse_args(argv)
    if not 0 <= opts.valid_fraction <= 0.5:
        parser.error("--valid-fraction must be in [0,0.5]")
    paths = [f for pat in opts.traces for f in sorted(glob.glob(pat))]
    stats = build(paths, opts.out, seed=opts.seed, val_fraction=opts.valid_fraction,
                  completed_only=not opts.include_incomplete)
    print(json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    main()
