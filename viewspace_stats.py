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


def _group_arrays(abs_n, pts_n, radii, eps):
    """Build the stat arrays for one subset of Gaussians (all inputs already masked)."""
    valid = torch.isfinite(abs_n) & torch.isfinite(pts_n) & (pts_n > eps)
    r = radii.float().clamp_min(1.0)
    arrays = {
        "abs": abs_n.float().cpu().numpy(),
        "points": pts_n.float().cpu().numpy(),
        "ratio": (abs_n[valid] / pts_n[valid]).float().cpu().numpy(),
        "radii": radii.float().cpu().numpy(),
        "abs_per_r": (abs_n / r).float().cpu().numpy(),
        "points_per_r": (pts_n / r).float().cpu().numpy(),
    }
    joint = {   # (points, ratio) pairs for the hexbin, restricted to valid ratio
        "points": pts_n[valid].float().cpu().numpy(),
        "ratio": arrays["ratio"],
    }
    return arrays, joint, int((~valid).sum())


@torch.no_grad()
def dump_viewspace_stats(split_pkg, iteration, out_dir, eps=1e-12, make_plot=True,
                         percent_of="group"):
    """Save |viewspace_abs|, |viewspace_points| and their ratio for visible Gaussians,
    split into gaussian_scene == 1 (flagged) and gaussian_scene == 0 (others).

    percent_of:
        "all"   -> y axis = % of ALL visible Gaussians (two groups together sum to 100%)
        "group" -> y axis = % within each group (each group sums to 100% by itself)

    Must be called after backward() and BEFORE optimizer.zero_grad().
    """
    assert percent_of in ("all", "group")
    os.makedirs(out_dir, exist_ok=True)

    visible = split_pkg["radii"] > 0
    flag_all = split_pkg["gaussian_scene"].reshape(-1) > 0.5

    abs_all = _screen_grad_norm(split_pkg["viewspace_abs"])
    pts_all = _screen_grad_norm(split_pkg["viewspace_points"])
    radii_all = split_pkg["radii"]

    groups = {
        "scene1": visible & flag_all,
        "scene0": visible & ~flag_all,
    }
    n_visible = int(visible.sum())

    data, joints = {}, {}
    print(f"\n[ITER {iteration}] viewspace stats  visible={n_visible}")
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for gname, gmask in groups.items():
        arrays, joint, n_zero = _group_arrays(abs_all[gmask], pts_all[gmask], radii_all[gmask], eps)
        data[gname], joints[gname] = arrays, joint
        n_g = int(gmask.sum())
        pct = 100.0 * n_g / max(n_visible, 1)
        print(f"  --- {gname}: n={n_g} ({pct:.2f}% of visible)  zero-grad (|points|<=eps)={n_zero}")
        for name, a in arrays.items():
            if a.size == 0:
                print(f"    {name:>12}: empty")
                continue
            qv = np.quantile(a, qs)
            print(f"    {name:>12}: mean={a.mean():.3e} std={a.std():.3e} | "
                  + " ".join(f"q{int(q*100):02d}={v:.3e}" for q, v in zip(qs, qv)))

    np.savez(
        os.path.join(out_dir, f"viewspace_stats_{iteration}.npz"),
        **{f"{g}_{k}": v for g, arrays in data.items() for k, v in arrays.items()},
    )

    if not make_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"scene1": "tab:red", "scene0": "tab:blue"}
    ylabel = "% of all visible" if percent_of == "all" else "% within group"

    def _weights(a_full_size, a_plot):
        # a_full_size: group size used as the percentage denominator
        denom = n_visible if percent_of == "all" else a_full_size
        return np.full(a_plot.shape, 100.0 / max(denom, 1))

    # ---- 1D histograms: both groups overlaid, shared log-spaced bins ----
    names = list(next(iter(data.values())).keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, name in zip(axes.ravel(), names):
        pos = [data[g][name][data[g][name] > 0] for g in groups]
        pos_all = np.concatenate([p for p in pos if p.size]) if any(p.size for p in pos) else np.array([])
        if pos_all.size == 0:
            ax.set_title(f"{name} (empty)")
            continue
        lo, hi = pos_all.min(), pos_all.max()
        if name == "radii":   # integer pixel radii -> one linear bin per integer
            bins = np.arange(np.floor(lo), np.ceil(hi) + 2) - 0.5
        else:
            bins = np.logspace(np.log10(lo), np.log10(hi if hi > lo else lo * 10), 100)
        for g, p in zip(groups, pos):
            if p.size == 0:
                continue
            # denominator = full group size (incl. zeros), so % reflects the whole group/all visible
            w = _weights(data[g][name].size, p)
            ax.hist(p, bins=bins, weights=w, histtype="step", lw=1.5, color=colors[g],
                    label=f"{g} (n={data[g][name].size}, med {np.median(p):.2e})")
            ax.axvline(np.median(p), color=colors[g], ls="--", lw=1)
        if name != "radii":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle(f"iteration {iteration}  (y: {ylabel})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"viewspace_stats_{iteration}.png"), dpi=120)
    plt.close(fig)

    # ---- Joint view per group: ratio vs |points|, colour = % of Gaussians per cell ----
    if any(joints[g]["ratio"].size for g in groups):
        allx = np.concatenate([joints[g]["points"] for g in groups if joints[g]["points"].size])
        ally = np.concatenate([joints[g]["ratio"] for g in groups if joints[g]["ratio"].size])
        allx, ally = allx[allx > 0], ally[ally > 0]
        extent = (np.log10(allx.min()), np.log10(allx.max()),
                  np.log10(ally.min()), np.log10(ally.max())) if allx.size and ally.size else None

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        hbs = []
        for ax, g in zip(axes, groups):
            x, y = joints[g]["points"], joints[g]["ratio"]
            keep = (x > 0) & (y > 0)
            x, y = x[keep], y[keep]
            if x.size == 0:
                ax.set_title(f"{g} (empty)")
                continue
            denom = n_visible if percent_of == "all" else data[g]["abs"].size
            hb = ax.hexbin(x, y, C=np.full(x.shape, 100.0 / max(denom, 1)),
                           reduce_C_function=np.sum, xscale="log", yscale="log",
                           bins="log", gridsize=80, extent=extent, mincnt=1)
            hbs.append((hb, ax))
            ax.set_xlabel("|viewspace_points.grad|")
            ax.set_ylabel("abs / points")
            ax.set_title(f"{g}  (n={data[g]['abs'].size})")
        # same colour scale for both groups so they can be compared directly
        vals = [hb.get_array() for hb, _ in hbs if hb.get_array().size]
        if vals:
            vmin = min(v[v > 0].min() for v in vals)
            vmax = max(v.max() for v in vals)
            for hb, ax in hbs:
                hb.set_clim(vmin, vmax)
                fig.colorbar(hb, ax=ax, label=ylabel)
        fig.suptitle(f"iteration {iteration}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"viewspace_joint_{iteration}.png"), dpi=120)
        plt.close(fig)