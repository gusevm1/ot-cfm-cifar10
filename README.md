# OT-CFM on CIFAR-10

A minimal, faithful implementation of **class-unconditional Optimal Transport Conditional Flow
Matching** on CIFAR-10, built on [`torchcfm`](https://github.com/atong01/conditional-flow-matching)
and [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq). Runs on CUDA, MPS or CPU.

📊 **[Results page →](https://gusevm1.github.io/ot-cfm-cifar10/)**

Training minimises `‖v_θ(x_t, t) − u_t‖²` where `(t, x_t, u_t)` come from an *exact minibatch OT*
coupling between Gaussian noise and data. Sampling integrates `dx/dt = v_θ(x, t)` from `t=0` to
`t=1` with an adaptive ODE solver.

## Setup

```bash
uv venv --python 3.12
uv pip install torch torchvision torchcfm torchdiffeq pot clean-fid tqdm
```

CIFAR-10 goes in `./data` (torchvision layout, i.e. `data/cifar-10-batches-py/`). Either symlink an
existing copy or fetch it once:

```bash
python -c "from torchvision import datasets; datasets.CIFAR10('data', download=True)"
```

## Usage

```bash
python test_sanity.py                      # spec §6.1 checks — run this first
python train.py --steps 50000 --batch_size 256 --out runs/base
python evaluate.py --ckpt runs/base/ckpt.pt --n 10000 --method dopri5 --fid
python export_results.py                   # refresh docs/results.js for the page
```

`train.py` auto-resumes from `runs/<name>/ckpt.pt` if it exists, so a killed run picks up where it
left off. It writes `metrics.jsonl`, an 8×8 sample grid every `--sample_every` steps, and a
checkpoint every `--ckpt_every`.

`evaluate.py --method euler --steps 20` swaps the adaptive solver for a fixed-step one — same
network, ~6× cheaper, near-identical images (see the solver-invariance check).

## Implementation ↔ spec

| Spec | Where |
|---|---|
| §1 U-Net, 64 base ch, mult (1,2,2,2), 2 res blocks, no class cond. | `otcfm.build_model` |
| §1 EMA, decay 0.9999 | `otcfm.build_ema` — `torch.optim.swa_utils.AveragedModel` |
| §2 AdamW, wd 1e-4, lr 2e-4, 500-step warmup → cosine | `train.py` |
| §2 RandomHorizontalFlip(0.5), values in [-1, 1] | `otcfm.cifar10` |
| §3 `sample_location_and_conditional_flow` + MSE | `train.py` main loop |
| §4 `odeint(..., method='dopri5')`, clamp → uint8 | `otcfm.sample`, `otcfm.to_uint8` |
| §6.1 overfit / solver invariance / OOM checks | `test_sanity.py` |
| §6.2 8×8 grid + 10k-sample clean-FID | `evaluate.py` |

**Parameter count.** The spec estimates ~11.5M for that configuration; `torchcfm`'s U-Net at
exactly those hyperparameters is **9.28M**. The knobs match the spec, the count is what it is.
`--base_channels 128` gives the 35.7M variant the spec's second FID target refers to.

## Notes

- **MPS float64.** Adaptive `torchdiffeq` solvers keep time/tolerance scalars in float64, which MPS
  doesn't have. `otcfm.sample` pins them to float32 on MPS only; CUDA/CPU keep the default.
- **Frozen OT coupling in the overfit test.** `torchcfm` redraws the OT plan *with replacement*
  every call, which duplicates pairs. The 16-image test freezes the coupling once
  (`pi.argmax`) so "integrate from x₀, get x₁ back" is a deterministic claim.
- **Gradient clipping** at 1.0 is on by default. Not in the spec; `--grad_clip 0` disables it.

## Reaching the spec's FID targets

The spec targets FID < 10 for the small U-Net. Measured here at 0.86 s/step, batch 256, M3 Max /
MPS / fp32: one epoch is ~2.8 min, so a 50k-step run is ~12 h ≈ 256 epochs. For reference,
`torchcfm`'s published CIFAR-10 OT-CFM number (FID 3.6) uses the 35.7M U-Net for 400k steps —
roughly 1000 epochs and 4× the parameters. **A local MPS run will not hit FID < 10**; see the
results page for what it does reach. The same commands with `--device cuda` on a single A100 close
that gap in about a day.

## License

MIT. `torchcfm` and `torchdiffeq` are the property of their respective authors.
