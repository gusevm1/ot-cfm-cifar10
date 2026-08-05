"""OT-CFM training (spec §3). Logs metrics.jsonl, dumps 8x8 sample grids, checkpoints for resume."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchvision.utils import save_image

from otcfm import build_ema, build_model, cifar10, device_auto, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--out", type=str, default="runs/base")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else device_auto()
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    model = build_model(args.base_channels).to(device)
    ema = build_ema(model, args.ema_decay).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)

    def lr_at(step):  # spec §2: linear warmup then cosine decay
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        prog = min(1.0, max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup)))
        return warm * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    start_step = 0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        print(f"resumed from {ckpt_path} @ step {start_step}", flush=True)

    loader = DataLoader(cifar10(train=True), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    steps_per_epoch = len(loader)

    cfg = {**vars(args), "device": str(device), "n_params": n_params,
           "steps_per_epoch": steps_per_epoch}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)

    log = open(out / "metrics.jsonl", "a")
    fixed_noise = torch.randn(64, 3, 32, 32, device=device, generator=torch.Generator(device).manual_seed(1234))

    step, t0, running = start_step, time.time(), 0.0
    while step < args.steps:
        for x1, _ in loader:
            if step >= args.steps:
                break
            model.train()
            x1 = x1.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)
            t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
            loss = torch.nn.functional.mse_loss(model(t, xt), ut)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:  # ponytail: not in the spec; cheap insurance for unattended runs
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            ema.update_parameters(model)

            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                rec = {"step": step, "epoch": step / steps_per_epoch,
                       "loss": running / args.log_every, "lr": sched.get_last_lr()[0],
                       "sec_per_step": (time.time() - t0) / args.log_every,
                       "wall_h": (time.time() - t0) / 3600 if step == args.log_every else None}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"step {step:>7} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                      f"{rec['sec_per_step']:.2f}s/step", flush=True)
                running, t0 = 0.0, time.time()

            if step % args.sample_every == 0:
                grid = sample(ema, 64, device, method="euler", steps=50, x0=fixed_noise)
                save_image(grid.clamp(-1, 1) * 0.5 + 0.5, out / "samples" / f"{step:07d}.png", nrow=8)
                t0 = time.time()

            if step % args.ckpt_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "config": cfg}, ckpt_path)
                t0 = time.time()

    log.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
