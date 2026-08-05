"""Spec §6.1 sanity checks. Run: python test_sanity.py [--quick]

1. 16-image overfit    — loss -> 0, integrating the matched x0 reproduces the 16 targets
2. solver invariance   — dopri5 vs fixed-step euler(20) agree
3. hardware / memory   — one full CIFAR-10 epoch at batch 256, no OOM
"""

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchvision.utils import save_image

from otcfm import build_model, cifar10, device_auto, sample, to_uint8

RESULTS = {}
OUT = "runs/sanity"


def save():
    """Written after every check so a later failure doesn't discard earlier results."""
    p = f"{OUT}/sanity.json"
    prev = json.load(open(p)) if __import__("os").path.exists(p) else {}
    json.dump({**prev, **RESULTS}, open(p, "w"), indent=2)


def overfit_16(device, steps=500, out="runs/sanity"):
    torch.manual_seed(0)
    x1 = torch.stack([cifar10(train=True, augment=False)[i][0] for i in range(16)]).to(device)
    x0 = torch.randn_like(x1)

    # Freeze the exact-OT coupling once so the reproduction check is deterministic.
    # torchcfm resamples the plan (with replacement) every call, which duplicates pairs.
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    pi = fm.ot_sampler.get_map(x0, x1)
    x1 = x1[pi.argmax(axis=1)]

    net = build_model().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    t0 = time.time()
    for s in range(steps):
        net.train()
        t = torch.rand(16, device=device)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        loss = torch.nn.functional.mse_loss(net(t, xt), x1 - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0:
            print(f"  step {s + 1:>4}  loss {loss.item():.5f}", flush=True)

    recon = sample(net, 16, device, method="dopri5", x0=x0)
    err = (recon.clamp(-1, 1) - x1).abs().mean().item()
    save_image(torch.cat([x1, recon.clamp(-1, 1)]) * 0.5 + 0.5, f"{out}/overfit16.png", nrow=8)
    RESULTS["overfit_16"] = {"final_loss": loss.item(), "recon_mae": err,
                             "steps": steps, "seconds": time.time() - t0}
    print(f"  final loss {loss.item():.5f}  recon MAE {err:.4f}  ({time.time() - t0:.0f}s)")
    save()
    assert loss.item() < 0.05, "overfit loss did not collapse"
    assert err < 0.10, "ODE round-trip did not reproduce the targets"
    return net


def solver_invariance(net, device, out="runs/sanity"):
    x0 = torch.randn(16, 3, 32, 32, device=device, generator=torch.Generator(device).manual_seed(7))
    a = sample(net, 16, device, method="dopri5", x0=x0)
    b = sample(net, 16, device, method="euler", steps=20, x0=x0)
    d = (to_uint8(a).float() - to_uint8(b).float()).abs()
    save_image(torch.cat([a, b]).clamp(-1, 1) * 0.5 + 0.5, f"{out}/solver_invariance.png", nrow=8)
    RESULTS["solver_invariance"] = {"mae_uint8": d.mean().item(), "max_uint8": d.max().item()}
    print(f"  dopri5 vs euler(20):  mean |Δ| {d.mean():.2f}/255   max {d.max():.0f}/255")
    save()
    assert d.mean().item() < 3.0, "solvers disagree beyond rounding"


def hardware_check(device, batch_size=256, max_steps=None):
    fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    net = build_model().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4)
    loader = DataLoader(cifar10(train=True), batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=True, persistent_workers=True)
    n = max_steps or len(loader)
    t0 = time.time()
    for i, (x1, _) in enumerate(loader):
        if i >= n:
            break
        x1 = x1.to(device)
        t, xt, ut = fm.sample_location_and_conditional_flow(torch.randn_like(x1), x1)
        loss = torch.nn.functional.mse_loss(net(t, xt), ut)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n} steps  {(time.time() - t0) / (i + 1):.2f}s/step", flush=True)
    sec = (time.time() - t0) / n
    RESULTS["hardware"] = {"batch_size": batch_size, "steps_run": n,
                           "steps_per_epoch": len(loader), "sec_per_step": sec,
                           "epoch_minutes": sec * len(loader) / 60,
                           "device": str(device), "oom": False}
    print(f"  {sec:.2f}s/step at batch {batch_size} -> {sec * len(loader) / 60:.1f} min/epoch")
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="20 steps instead of a full epoch in check 3")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="runs/sanity")
    p.add_argument("--only", type=int, nargs="*", choices=[1, 2, 3], help="run a subset of checks")
    a = p.parse_args()
    dev = torch.device(a.device) if a.device else device_auto()
    OUT = a.out
    only = a.only or [1, 2, 3]
    __import__("pathlib").Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"device: {dev}")
    net = None
    if 1 in only or 2 in only:
        print("[1/3] 16-image overfit")
        net = overfit_16(dev, out=a.out)
    if 2 in only:
        print("[2/3] solver invariance")
        solver_invariance(net, dev, out=a.out)
    if 3 in only:
        print(f"[3/3] hardware / memory ({'20 steps' if a.quick else 'full epoch'} @ batch 256)")
        hardware_check(dev, max_steps=20 if a.quick else None)
    print("\nall checks passed ->", f"{a.out}/sanity.json")
