"""Spec §6.2: generate N samples with the EMA net via odeint, save an 8x8 grid, compute clean-FID."""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import certifi
import torch
from PIL import Image
from torchvision.utils import save_image

from otcfm import build_ema, build_model, device_auto, sample, to_uint8


def patch_sqrtm():
    """clean-fid calls scipy.linalg.sqrtm(..., disp=False); scipy dropped `disp` in 1.16.

    Restoring the old two-value signature keeps the repo working on current scipy instead of
    pinning every user to a four-year-old release. clean-fid discards the error estimate.
    """
    from scipy import linalg

    if "disp" in inspect.signature(linalg.sqrtm).parameters:
        return
    real = linalg.sqrtm

    def sqrtm(A, disp=True, **kw):
        out = real(A)
        return out if disp else (out, float("nan"))

    linalg.sqrtm = sqrtm


def load_ema(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["config"]["base_channels"]).to(device)
    ema = build_ema(model, ck["config"]["ema_decay"]).to(device)
    ema.load_state_dict(ck["ema"])
    return ema, ck["step"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/base/ckpt.pt")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--method", type=str, default="dopri5")
    p.add_argument("--steps", type=int, default=None, help="fixed-step count for euler/rk4")
    p.add_argument("--out", type=str, default=None, help="defaults to <ckpt dir>/eval")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--fid", action="store_true", help="compute clean-FID after sampling")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else device_auto()
    ema, step = load_ema(args.ckpt, device)
    out = Path(args.out or Path(args.ckpt).parent / "eval")
    img_dir = out / f"{args.method}{args.steps or ''}_{args.n}"
    img_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator(device).manual_seed(args.seed)
    t0, i = time.time(), 0
    while i < args.n:
        b = min(args.batch_size, args.n - i)
        x = sample(ema, b, device, method=args.method, steps=args.steps, generator=g)
        if i == 0:
            save_image(x[:64].clamp(-1, 1) * 0.5 + 0.5, out / f"grid_{args.method}.png", nrow=8)
        for j, img in enumerate(to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()):
            Image.fromarray(img).save(img_dir / f"{i + j:06d}.png")
        i += b
        print(f"{i}/{args.n}  {(time.time() - t0) / i:.3f}s/img", flush=True)

    res = {"ckpt": args.ckpt, "step": step, "n": args.n, "method": args.method,
           "solver_steps": args.steps, "sample_seconds": time.time() - t0}

    if args.fid:
        # clean-fid downloads its Inception weights and reference stats over HTTPS. A python.org
        # Python has no cert bundle of its own, so point it at certifi's unless the caller set one.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        patch_sqrtm()
        from cleanfid import fid as cleanfid
        # clean-fid's resizer is a closure, so its loader workers can't be pickled for macOS
        # spawn. Single-process loading costs about a minute per 10k images.
        workers = 0 if sys.platform == "darwin" else 12
        res["fid"] = cleanfid.compute_fid(str(img_dir), dataset_name="cifar10", dataset_res=32,
                                          dataset_split=args.split, mode="clean", device=device,
                                          num_workers=workers)
        res["fid_ref"] = f"cifar10-{args.split}"
        print(f"FID({args.split}) = {res['fid']:.3f}", flush=True)

    (out / f"result_{args.method}{args.steps or ''}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
