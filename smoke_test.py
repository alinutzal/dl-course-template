"""Course-environment smoke test for CSCI 4852/6952 Deep Learning (REC-09).

Run after installing the environment:

    python smoke_test.py

Exit code 0 means every required check passed. Optional checks (GPU, PyG,
datasets) are reported but only fail the run when the package is installed
and broken.
"""

import importlib
import platform
import sys
import time

REQUIRED = {
    "torch": "2.14",
    "torchvision": "0.29",
    "numpy": "2.",
    "pandas": "3.",
    "matplotlib": "3.",
    "sklearn": "1.",
    "yaml": "6.",
}
OPTIONAL = ["torch_geometric", "datasets", "jupyterlab", "nbclient", "pytest"]

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def check_versions():
    major, minor = sys.version_info[:2]
    record("python 3.12", (major, minor) == (3, 12), platform.python_version())
    for mod, prefix in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            record(f"{mod} {prefix}*", v.startswith(prefix), v)
        except Exception as e:  # noqa: BLE001 - report any import failure
            record(f"{mod} import", False, repr(e))


def make_data(torch, n=512, d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g)
    y = (x @ w > 0).long()
    return x, y


def train_once(torch, device, seed=123, epochs=150):
    """Tiny MLP on synthetic data; returns final loss and accuracy."""
    torch.manual_seed(seed)
    x, y = make_data(torch)
    x, y = x.to(device), y.to(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return loss.item(), acc


def check_training():
    import torch

    t0 = time.perf_counter()
    loss, acc = train_once(torch, "cpu")
    record("cpu training", acc > 0.8, f"loss={loss:.4f} acc={acc:.3f} ({time.perf_counter() - t0:.2f}s)")

    loss2, _ = train_once(torch, "cpu")
    record("cpu determinism (same seed -> same loss)", loss == loss2, f"{loss:.8f} vs {loss2:.8f}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch_ok = f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list()
        record("cuda device supported by this build", arch_ok,
               f"{name}, sm_{cap[0]}{cap[1]}, build archs={torch.cuda.get_arch_list()}")
        t0 = time.perf_counter()
        loss, acc = train_once(torch, "cuda")
        torch.cuda.synchronize()
        record("cuda training", acc > 0.8, f"loss={loss:.4f} acc={acc:.3f} ({time.perf_counter() - t0:.2f}s)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        loss, acc = train_once(torch, "mps")
        record("mps training (Apple silicon)", acc > 0.8, f"loss={loss:.4f} acc={acc:.3f}")
    else:
        print("[INFO] no GPU detected - CPU-only environment (fine for Labs 1-2; use OSC for Labs 3-5)")


def check_torchvision():
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=None).eval()  # current API; no download
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    record("torchvision resnet18 forward", tuple(out.shape) == (2, 1000), str(tuple(out.shape)))


def check_optional():
    for mod in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            print(f"[INFO] {mod} {getattr(m, '__version__', '')} available")
        except ImportError:
            print(f"[INFO] {mod} not installed (optional)")
            continue
        if mod == "torch_geometric":
            import torch
            from torch_geometric.nn import GCNConv

            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
            out = GCNConv(4, 2)(torch.randn(3, 4), edge_index)
            record("torch_geometric GCNConv forward", tuple(out.shape) == (3, 2), str(tuple(out.shape)))


if __name__ == "__main__":
    print(f"CSCI 4852/6952 environment check - {platform.platform()}\n")
    check_versions()
    check_training()
    check_torchvision()
    check_optional()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    sys.exit(1 if failed else 0)
