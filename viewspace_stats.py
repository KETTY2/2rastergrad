import os
import numpy as np
import torch


def _screen_grad_norm(t):
    """Per-Gaussian 2D norm of a viewspace tensor's gradient (falls back to the tensor itself)."""
    g = t.grad if getattr(t, "grad", None) is not None else t
    g = g.detach()
    if g.ndim == 1 or g.shape[-1] == 1:          # already a scalar per Gaussian
        return g.reshape(-1).abs()
    return torch.linalg.vector_norm(g[:, :2], dim=-1)


@torch.no_grad()
def dump_viewspace_stats(split_pkg, iteration, out_dir, eps=1e-12, make_plot=True):
    """Save |viewspace_abs|, |viewspace_points| and their ratio for visible Gaussians.

    Must be called after backward() and BEFORE optimizer.zero_grad().
    """
    os.makedirs(out_dir, exist_ok=True)

    visible = split_pkg["radii"] > 0
    abs_n = _screen_grad_norm(split_pkg["viewspace_abs"])[visible]
    pts_n = _screen_grad_norm(split_pkg["viewspace_points"])[visible]

    valid = torch.isfinite(abs_n) & torch.isfinite(pts_n) & (pts_n > eps)
    ratio = abs_n[valid] / pts_n[valid]

    arrays = {
        "abs": abs_n.float().cpu().numpy(),
        "points": pts_n.float().cpu().numpy(),
        "ratio": ratio.float().cpu().numpy(),
    }

    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    print(f"\n[ITER {iteration}] viewspace stats  visible={int(visible.sum())}  "
          f"zero-grad (|points|<=eps)={int((~valid).sum())}")
    for name, a in arrays.items():
        if a.size == 0:
            print(f"  {name:>7}: empty")
            continue
        qv = np.quantile(a, qs)
        print(f"  {name:>7}: mean={a.mean():.3e} std={a.std():.3e} | "
              + " ".join(f"q{int(q*100):02d}={v:.3e}" for q, v in zip(qs, qv)))

    np.savez(os.path.join(out_dir, f"viewspace_stats_{iteration}.npz"), **arrays)

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, a) in zip(axes, arrays.items()):
            a = a[a > 0]
            if a.size == 0:
                ax.set_title(f"{name} (empty)")
                continue
            bins = np.logspace(np.log10(a.min()), np.log10(a.max()), 100)
            ax.hist(a, bins=bins)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvline(np.median(a), color="k", ls="--", lw=1, label=f"median {np.median(a):.2e}")
            ax.set_title(f"{name}  (n={a.size})")
            ax.legend()
        fig.suptitle(f"iteration {iteration}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"viewspace_stats_{iteration}.png"), dpi=120)
        plt.close(fig)

        # Joint view: how the ratio depends on the plain gradient magnitude
        m = valid.cpu().numpy()
        if m.any():
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hexbin(arrays["points"][m], arrays["ratio"], xscale="log", yscale="log",
                      bins="log", gridsize=80)
            ax.set_xlabel("|viewspace_points.grad|")
            ax.set_ylabel("abs / points")
            ax.set_title(f"iteration {iteration}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"viewspace_joint_{iteration}.png"), dpi=120)
            plt.close(fig)
