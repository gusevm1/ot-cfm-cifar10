"""OT-CFM on CIFAR-10: model, data, EMA, sampling. Shared by train.py / evaluate.py / test_sanity.py."""

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdiffeq import odeint
from torchvision import datasets, transforms

DATA_ROOT = "data"


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(base_channels=64):
    """Spec §1: time-conditioned 2D U-Net, mult (1,2,2,2), 2 res blocks, class-unconditional."""
    return UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=base_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16,8",
        dropout=0.1,
    )


def build_ema(model, decay=0.9999):
    """Spec §1: shadow copy, EMA decay 0.9999. torch.optim.swa_utils already does this."""
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def cifar10(train=True, augment=True):
    """Spec §2: RandomHorizontalFlip(0.5) on train, values mapped to [-1, 1]."""
    tf = [transforms.RandomHorizontalFlip(0.5)] if (train and augment) else []
    # Normalize, not a lambda: DataLoader workers spawn on macOS and lambdas don't pickle.
    tf += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return datasets.CIFAR10(DATA_ROOT, train=train, download=False, transform=transforms.Compose(tf))


@torch.no_grad()
def sample(net, n, device, method="dopri5", steps=None, x0=None, generator=None):
    """Spec §4: integrate dx/dt = v(x, t) from t=0 to t=1, return float tensor in [-1, 1].

    `steps` sets a fixed-step grid (used by euler/midpoint/rk4); dopri5 ignores it and
    adapts. Returns the state at t=1 only.
    """
    net.eval()
    if x0 is None:
        x0 = torch.randn(n, 3, 32, 32, device=device, generator=generator)

    def f(t, x):
        return net(t.expand(x.shape[0]), x)

    t_span = torch.linspace(0, 1, (steps or 1) + 1, device=device)
    opts = {}
    if method in ("dopri5", "dopri8", "bosh3", "adaptive_heun"):
        t_span = torch.tensor([0.0, 1.0], device=device)
        # MPS has no float64; adaptive solvers keep their time/tolerance scalars in fp64 by
        # default. fp32 step control is well inside a 1e-5 tolerance for 8-bit image output.
        if device.type == "mps":
            opts["dtype"] = torch.float32
    traj = odeint(f, x0, t_span, method=method, atol=1e-5, rtol=1e-5, options=opts)
    return traj[-1]


def to_uint8(x):
    """Spec §2/§4: clamp to [-1, 1], map to [0, 255] uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
