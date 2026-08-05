"""Track FID while training runs: poll the checkpoint, evaluate it, append to fid.jsonl.

Deliberately a thin poller around evaluate.py rather than a hook inside train.py — it can
be started, killed and restarted without touching the run.

The default 2k samples is far below the 10k the spec asks for; FID is biased upward at small
sample counts, so treat this series as a *trend*, not a score. `evaluate.py --n 10000` is the
number to quote.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def current_step(ckpt):
    try:
        return torch.load(ckpt, map_location="cpu", weights_only=False)["step"]
    except Exception:
        return None  # mid-write; try again next poll


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/base/ckpt.pt")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--method", type=str, default="euler")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--every", type=int, default=2000, help="minimum training steps between evals")
    p.add_argument("--poll", type=int, default=120, help="seconds between checkpoint polls")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    log = ckpt.parent / "fid.jsonl"
    done = {json.loads(l)["step"] for l in log.read_text().splitlines() if l.strip()} if log.exists() else set()
    last = max(done, default=-args.every)

    while True:
        step = current_step(ckpt) if ckpt.exists() else None
        if step is None or step - last < args.every:
            time.sleep(args.poll)
            continue

        tmp = ckpt.parent / "fid_tmp"
        cmd = [sys.executable, "evaluate.py", "--ckpt", str(ckpt), "--n", str(args.n),
               "--method", args.method, "--fid", "--out", str(tmp)]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        if args.device:
            cmd += ["--device", args.device]

        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"eval failed at step {step}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}", flush=True)
            time.sleep(args.poll)
            continue

        res = json.loads((tmp / f"result_{args.method}{args.steps or ''}.json").read_text())
        rec = {"step": step, "fid": res["fid"], "n": args.n, "method": args.method,
               "solver_steps": args.steps, "seconds": time.time() - t0}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"step {step:>7}  FID-{args.n // 1000}k {res['fid']:.2f}  ({rec['seconds'] / 60:.1f} min)", flush=True)
        last = step


if __name__ == "__main__":
    main()
