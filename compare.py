"""Generated samples next to their nearest real CIFAR-10 image, at one checkpoint.

The model is unconditional, so no generated image has a paired ground truth. The closest
thing is retrieval: for each sample, the nearest real training image. Two things to read
off it — samples should get *closer* to the real manifold as training proceeds, and if the
distance collapses toward zero the model is memorising rather than generating.

Distances are plain pixel-space L2 on [-1, 1] images. Cheap and standard for this check;
it keys on colour and layout more than semantics, so read the trend, not the absolute value.
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from evaluate import load_ema
from otcfm import cifar10, device_auto, sample

CHUNK = 5000  # rows of CIFAR-10 moved to the device at a time


def nearest_real(x, device, train, k=1):
    """L2 nearest neighbours of each row of `x` among `train`. Returns (dist, index)."""
    q = x.flatten(1)
    best_d = torch.full((q.shape[0],), float("inf"), device=device)
    best_i = torch.zeros(q.shape[0], dtype=torch.long, device=device)
    for s in range(0, train.shape[0], CHUNK):
        ref = train[s:s + CHUNK].to(device, torch.float32).flatten(1).div_(127.5).sub_(1.0)
        d = torch.cdist(q, ref)
        dm, im = d.min(dim=1)
        hit = dm < best_d
        best_i[hit] = im[hit] + s
        best_d[hit] = dm[hit]
    return best_d, best_i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/base/ckpt.pt")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--method", type=str, default="euler")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", type=str, default=None, help="defaults to <ckpt dir>/compare")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234, help="same default as train.py's grid noise")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else device_auto()
    ema, step = load_ema(args.ckpt, device)
    out = Path(args.out or Path(args.ckpt).parent / "compare")
    out.mkdir(parents=True, exist_ok=True)

    # Same noise at every checkpoint, so the strip is comparable across steps.
    x0 = torch.randn(args.n, 3, 32, 32, device=device,
                     generator=torch.Generator(device).manual_seed(args.seed))
    gen = sample(ema, args.n, device, method=args.method, steps=args.steps, x0=x0).clamp(-1, 1)

    ds = cifar10(train=True, augment=False)
    train = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,32,32)
    dist, idx = nearest_real(gen, device, train)
    real = train[idx.cpu()].to(device, torch.float32).div_(127.5).sub_(1.0)

    # Rows of 8, alternating generated / nearest-real so each column is a pair.
    strips = []
    for s in range(0, args.n, 8):
        strips += [gen[s:s + 8], real[s:s + 8]]
    save_image(torch.cat(strips) * 0.5 + 0.5, out / f"{step:07d}_nn.png", nrow=8)

    rec = {"step": step, "n": args.n, "src": f"compare/{step:07d}_nn.png",
           "nn_mean": dist.mean().item(), "nn_min": dist.min().item(), "nn_max": dist.max().item()}
    with open(out / "compare.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

    # Static reference: what real CIFAR-10 actually looks like, same 8-wide layout.
    ref = out / "real_reference.png"
    if not ref.exists():
        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(train.shape[0], generator=g)[:64]
        save_image(train[pick].float().div(255.0), ref, nrow=8)

    print(f"step {step:>7}  nn L2 mean {rec['nn_mean']:.2f}  min {rec['nn_min']:.2f}  -> {rec['src']}")


if __name__ == "__main__":
    main()
