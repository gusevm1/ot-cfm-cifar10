"""How few solver steps can this flow get away with?

OT-CFM's selling point is straight probability paths, so quality should hold as the
step count drops. Renders the same fixed noise batch at several Euler budgets plus the
adaptive dopri5 reference, and reports each one's error against that reference.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from evaluate import load_ema
from otcfm import device_auto, sample, to_uint8

BUDGETS = [1, 2, 4, 8, 16, 32, 50]


def count_nfe(net):
    """dopri5 picks its own step count; wrap forward to see what it actually cost."""
    n = [0]
    fwd = net.forward

    def counting(*a, **kw):
        n[0] += 1
        return fwd(*a, **kw)

    net.forward = counting
    return n, lambda: setattr(net, "forward", fwd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/base/ckpt.pt")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", type=str, default=None, help="defaults to <ckpt dir>/solver")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else device_auto()
    ema, step = load_ema(args.ckpt, device)
    out = Path(args.out or Path(args.ckpt).parent / "solver")
    out.mkdir(parents=True, exist_ok=True)

    x0 = torch.randn(args.n, 3, 32, 32, device=device,
                     generator=torch.Generator(device).manual_seed(args.seed))

    nfe, restore = count_nfe(ema)
    t0 = time.time()
    ref = sample(ema, args.n, device, method="dopri5", x0=x0)
    entries = [{"method": "dopri5", "steps": None, "nfe": nfe[0], "seconds": time.time() - t0,
                "mae_vs_dopri5": 0.0, "src": "solver/dopri5.png"}]
    restore()
    save_image(ref.clamp(-1, 1) * 0.5 + 0.5, out / "dopri5.png", nrow=8)
    print(f"dopri5   nfe {nfe[0]:>4}  {entries[0]['seconds']:.1f}s")

    for b in BUDGETS:
        t0 = time.time()
        x = sample(ema, args.n, device, method="euler", steps=b, x0=x0)
        mae = (to_uint8(x).float() - to_uint8(ref).float()).abs().mean().item()
        save_image(x.clamp(-1, 1) * 0.5 + 0.5, out / f"euler{b}.png", nrow=8)
        entries.append({"method": "euler", "steps": b, "nfe": b, "seconds": time.time() - t0,
                        "mae_vs_dopri5": mae, "src": f"solver/euler{b}.png"})
        print(f"euler{b:<3} nfe {b:>4}  {time.time() - t0:.1f}s  MAE vs dopri5 {mae:.2f}/255")

    (out / "solver.json").write_text(json.dumps({"step": step, "n": args.n, "entries": entries}, indent=2))
    print("->", out / "solver.json")


if __name__ == "__main__":
    main()
