"""Track a run's quality while it trains: poll the checkpoint, measure, append to jsonl.

Deliberately a poller around evaluate.py / compare.py rather than hooks inside train.py — it
can be started, killed and restarted without touching the run, and it never risks the training
process. Every new checkpoint gets a nearest-neighbour comparison (seconds); FID is slower, so
it runs on its own step interval.

The FID sample count here (2k by default) is far below the 10k the spec asks for, and FID is
biased upward at small counts. This series is a trend, not a score — `evaluate.py --n 10000`
produces the number to quote.
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
        return None  # mid-write, or not there yet; try again next poll


def last_step(path):
    if not path.exists():
        return None
    steps = [json.loads(l)["step"] for l in path.read_text().splitlines() if l.strip()]
    return max(steps, default=None)


def run(cmd, label, step):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{label} failed at step {step}:\n{r.stdout[-600:]}\n{r.stderr[-600:]}", flush=True)
        return None
    return r.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/base/ckpt.pt")
    p.add_argument("--n", type=int, default=2000, help="samples per FID measurement")
    p.add_argument("--method", type=str, default="euler")
    p.add_argument("--steps", type=int, default=20, help="solver steps for the sampler")
    p.add_argument("--fid_every", type=int, default=2000, help="min training steps between FID evals")
    p.add_argument("--poll", type=int, default=60, help="seconds between checkpoint polls")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    fid_log = ckpt.parent / "fid.jsonl"
    dev = ["--device", args.device] if args.device else []

    seen = last_step(ckpt.parent / "compare" / "compare.jsonl")
    seen = seen if seen is not None else -1
    last_fid = last_step(fid_log)
    last_fid = last_fid if last_fid is not None else -args.fid_every

    while True:
        step = current_step(ckpt) if ckpt.exists() else None
        if step is None or step <= seen:
            time.sleep(args.poll)
            continue

        out = run([sys.executable, "compare.py", "--ckpt", str(ckpt),
                   "--method", args.method, "--steps", str(args.steps)] + dev, "compare", step)
        if out:
            print(out.strip(), flush=True)
            seen = step

        if step - last_fid >= args.fid_every:
            tmp = ckpt.parent / "fid_tmp"
            t0 = time.time()
            ok = run([sys.executable, "evaluate.py", "--ckpt", str(ckpt), "--n", str(args.n),
                      "--method", args.method, "--steps", str(args.steps), "--fid",
                      "--out", str(tmp)] + dev, "fid", step)
            if ok:
                res = json.loads((tmp / f"result_{args.method}{args.steps}.json").read_text())
                rec = {"step": step, "fid": res["fid"], "n": args.n, "method": args.method,
                       "solver_steps": args.steps, "seconds": time.time() - t0}
                with open(fid_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"step {step:>7}  FID-{args.n // 1000}k {res['fid']:.2f} "
                      f"({rec['seconds'] / 60:.1f} min)", flush=True)
                last_fid = step

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
