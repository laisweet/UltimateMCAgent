import json
import sys
import tempfile
import unittest
from pathlib import Path

from prepare_sft import build, load_episode, scrub
from evaluate_offline import evaluate

class DatasetTest(unittest.TestCase):
    def test_scrub(self):
        src = {"api_key":"secret", "status":{"pos":[1,2,3]}, "events":[{"text":"private"}]}
        self.assertEqual(scrub(src), {"status":{"pos":[1,2,3]}})

    def test_episode_split_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i in range(3):
                path = Path(temp) / f"run{i}.jsonl"
                path.write_text("\n".join([
                    json.dumps({"type":"trace", "step":1,"error":None,
                                "observation":{"status":{"pos":[i,64,0]}},
                                "decision":{"action":"wait","args":{"seconds":1},"reason":"observe"},
                                "result":{"waited":1}}),
                    json.dumps({"type":"complete","step":2})
                ]))
                paths.append(str(path))
            out = Path(temp) / "dataset"
            result = build(paths,out,seed=5,val_fraction=.33)
            self.assertEqual(result["train"]["examples"], 2)
            self.assertEqual(result["valid"]["examples"], 1)
            records = [json.loads(x) for x in (out/"train.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["messages"][2]["role"],"assistant")

    def test_teacher_agreement(self):
        target = {"action":"wait","args":{"seconds":1}, "reason":"observe"}
        row = {"messages":[{"role":"system","content":"a"}, {"role":"user","content":"b"},
                          {"role":"assistant","content":json.dumps(target)}]}
        metrics, _ = evaluate([row], lambda _: json.dumps(target))
        self.assertEqual(metrics["exact_args_accuracy"],1.0)

if __name__ == "__main__":
    unittest.main()
