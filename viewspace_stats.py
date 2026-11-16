import os
import re
import numpy as np
import torch


# 내부 키 -> 그래프에 표시할 이름 (npz 키는 기존 그대로 유지)
DISPLAY_NAMES = {
    "abs": "ABS_Grad",
    "points": "Grad",
    "ratio": "ABS_Grad / Grad",
}


def _disp(name):
    return DISPLAY_NAMES.get(name, name)


def _img_title(tag, img_id=None):
    """그래프 제목용 라벨. iter/pass는 빼고 이미지 번호만 남긴다.

        "030000_pass1_img0007" -> "IMG0007"
        img_id=7               -> "IMG7"       (명시하면 그 값을 그대로 사용)
        "foo"                  -> "IMGfoo"     (img#### 패턴이 없을 때의 fallback)
    """
    if img_id is not None:
        return f"IMG{img_id}"
    m = re.search(r"img(\d+)", str(tag), flags=re.IGNORECASE)
    if m:
        return f"IMG{m.group(1)}"
    return f"IMG{tag}"


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
    arrays = {
        "abs": abs_n.float().cpu().numpy(),
        "points": pts_n.float().cpu().numpy(),
        "ratio": (abs_n[valid] / pts_n[valid]).float().cpu().numpy(),
    }
    return arrays, int((~valid).sum())


def _value_bins(v, n_bins=50):
    """Linear bins: one bin per integer if the values are integer-like, else n_bins equal bins."""
    lo, hi = float(v.min()), float(v.max())
    if np.all(v == np.round(v)) and hi - lo <= 200:
        return np.arange(np.floor(lo), np.ceil(hi) + 2) - 0.5
    return np.linspace(lo, hi if hi > lo else lo + 1.0, n_bins + 1)


def _plot_scene_num(plt, values, colors, value_name, title, path):
    """One histogram panel per entry of `values` (name -> 1D array), shared bins.
    y axis = % within that panel's own set of Gaussians."""
    non_empty = [v for v in values.values() if v.size]
    if not non_empty:
        return
    bins = _value_bins(np.concatenate(non_empty))
    fig, axes = plt.subplots(1, len(values), figsize=(5.4 * len(values), 4.5), sharex=True)
    for ax, (name, v) in zip(np.atleast_1d(axes), values.items()):
        if v.size == 0:
            ax.set_title(f"{name} (empty)")
            continue
        ax.hist(v, bins=bins, weights=np.full(v.shape, 100.0 / v.size),
                color=colors.get(name, "tab:gray"), alpha=0.75,
                label=f"n={v.size}\nmean {v.mean():.3g}\nmed {np.median(v):.3g}\n"
                      f"min {v.min():.3g} / max {v.max():.3g}")
        ax.set_yscale("log")
        ax.set_xlabel(value_name)
        ax.set_ylabel(f"% of {name}")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_counts(plt, counts, title, path, ylabel="# Gaussians"):
    """Bar chart of Gaussian counts; each bar is annotated with its value and % of the first bar."""
    names, nums = list(counts.keys()), [float(v) for v in counts.values()]
    total = max(nums[0], 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(names, nums, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"][:len(names)])
    for b, n in zip(bars, nums):
        ax.annotate(f"{n:,.1f}\n({100.0 * n / total:.2f}%)", (b.get_x() + b.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(nums) * 1.18 if max(nums) > 0 else 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_mean_counts(n_total, scene_num, total_visible_radii, num_cameras, out_dir,
                     tag="", make_plot=True):
    """카운트 pass가 끝난 뒤 한 번만 호출: view 평균 개수 비교 막대 그래프.

        total  = 전체 Gaussian 개수
        radii  = total_visible_radii / num_cameras          (view당 평균 radii > 0 개수)
        render = sum(pc._gaussian_scene_num) / num_cameras  (view당 평균 gaussian_adam 누적값)

    Outputs (out_dir): viewspace_counts[_{tag}].png / .npz
    """
    os.makedirs(out_dir, exist_ok=True)
    n_cam = max(int(num_cameras), 1)
    counts = {
        "total": float(n_total),
        "radii": float(total_visible_radii) / n_cam,
        "render": float(scene_num.detach().float().sum().item()) / n_cam,
    }
    print("\n[viewspace counts] mean per view over "
          f"{n_cam} cameras: " + ", ".join(f"{k}={v:,.1f}" for k, v in counts.items()))

    suffix = f"_{tag}" if tag != "" else ""
    np.savez(os.path.join(out_dir, f"viewspace_counts{suffix}.npz"),
             count_names=np.array(list(counts.keys())),
             count_values=np.array(list(counts.values())))
    if not make_plot:
        return counts

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_counts(
        plt, counts, f"Gaussian counts (mean per view, {n_cam} cameras)",
        os.path.join(out_dir, f"viewspace_counts{suffix}.png"),
        ylabel="# Gaussians (mean per view)",
    )
    return counts


@torch.no_grad()
def dump_viewspace_stats(split_pkg, iteration, out_dir, eps=1e-12, make_plot=True,
                         percent_of="group", scene_num=None, img_id=None):
    """Save |viewspace_abs|, |viewspace_points| and their ratio for visible Gaussians,
    split into gaussian_scene == 1 (flagged) and gaussian_scene == 0 (others),
    plus combined visible Gaussians.

    Outputs (all in out_dir):
        viewspace_stats_{it}.npz / .png   scene1 vs scene0 histograms of abs, points, ratio
        viewspace_stats_all_{it}.png      same, all visible Gaussians together
        viewspace_scene_num_{it}.png      distribution of pc._gaussian_scene_num
                                          for scene0 / scene1 / all visible

    scene_num: per-Gaussian tensor pc._gaussian_scene_num, taken at render time
               (same length / order as split_pkg["radii"]). None -> 이 히스토그램은 생략.
    (개수 비교 막대 그래프는 이미지별로 그리지 않는다 -> plot_mean_counts 참고)

    percent_of:
        "all"   -> y axis = % of ALL visible Gaussians (two groups together sum to 100%)
        "group" -> y axis = % within each group (each group sums to 100% by itself)

    Must be called after backward() and BEFORE optimizer.zero_grad().
    """
    assert percent_of in ("all", "group")
    os.makedirs(out_dir, exist_ok=True)

    visible = split_pkg["gaussian_adam"].reshape(-1) > 0
    flag_all = split_pkg["gaussian_scene"].reshape(-1) > 0.5

    abs_all = _screen_grad_norm(split_pkg["viewspace_abs"])
    pts_all = _screen_grad_norm(split_pkg["viewspace_points"])
    radii_all = split_pkg["radii"]

    groups = {
        "High-Error": visible & flag_all,
        "Normal": visible & ~flag_all,
    }
    n_total = int(visible.numel())
    n_visible = int(visible.sum())

    data = {}
    print(f"\n[ITER {iteration}] viewspace stats  visible={n_visible}")
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for gname, gmask in groups.items():
        arrays, n_zero = _group_arrays(abs_all[gmask], pts_all[gmask], radii_all[gmask], eps)
        data[gname] = arrays
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

    # 전체 visible 가우시안 통계 계산
    all_arrays, _ = _group_arrays(abs_all[visible], pts_all[visible], radii_all[visible], eps)

    # ---- pc._gaussian_scene_num 값 분포 (scene0 / scene1 / 전체 visible) ----
    scene_key = "gaussian_scene_num"
    scene_vals = {}
    if scene_num is None:
        print(f"  [warn] scene_num not given; skipping the {scene_key} histogram")
    else:
        scene_val_all = scene_num.detach().reshape(-1).float()
        if scene_val_all.numel() == n_total:
            scene_vals = {
                "Normal": scene_val_all[groups["Normal"]].cpu().numpy(),
                "High-Error": scene_val_all[groups["High-Error"]].cpu().numpy(),
                "All": scene_val_all[visible].cpu().numpy(),
            }
            for name, v in scene_vals.items():
                if v.size:
                    print(f"  {scene_key} [{name}]: n={v.size} mean={v.mean():.3g} "
                          f"min={v.min():.3g} med={np.median(v):.3g} max={v.max():.3g}")
        else:
            print(f"  [warn] scene_num has {scene_val_all.numel()} values, "
                  f"expected {n_total}; skipping its histogram")

    np.savez(
        os.path.join(out_dir, f"viewspace_stats_{iteration}.npz"),
        **{f"{g}_{k}": v for g, arrays in data.items() for k, v in arrays.items()},
        **{f"all_{k}": v for k, v in all_arrays.items()},
        **{f"scene_num_{k.replace(' ', '_')}": v for k, v in scene_vals.items()},
    )

    if not make_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"High-Error": "tab:red", "Normal": "tab:blue", "All": "tab:green"}
    ylabel = "% of All" if percent_of == "all" else "% within group"
    fig_title = _img_title(iteration, img_id)   # 파일명은 iteration(tag) 그대로, 제목만 IMG####

    def _weights(a_full_size, a_plot):
        denom = n_visible if percent_of == "all" else a_full_size
        return np.full(a_plot.shape, 100.0 / max(denom, 1))

    names = list(next(iter(data.values())).keys())

    # ---- 시각화 1: scene1 vs scene0 비교 히스토그램 (1x3 레이아웃) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, name in zip(axes.ravel(), names):
        pos = [data[g][name][data[g][name] > 0] for g in groups]
        pos_all = np.concatenate([p for p in pos if p.size]) if any(p.size for p in pos) else np.array([])
        if pos_all.size == 0:
            ax.set_title(f"{_disp(name)} (empty)")
            continue
        lo, hi = pos_all.min(), pos_all.max()
        bins = np.logspace(np.log10(lo), np.log10(hi if hi > lo else lo * 10), 100)
        for g, p in zip(groups, pos):
            if p.size == 0:
                continue
            w = _weights(data[g][name].size, p)
            ax.hist(p, bins=bins, weights=w, histtype="step", lw=1.5, color=colors[g],
                    label=f"{g} (n={data[g][name].size}, med {np.median(p):.2e})")
            ax.axvline(np.median(p), color=colors[g], ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(_disp(name))
        ax.legend(fontsize=8)
    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"viewspace_stats_{iteration}.png"), dpi=120)
    plt.close(fig)

    # ---- 시각화 2: scene 구분 없이 전체 visible 가우시안 히스토그램 (1x3 레이아웃) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, name in zip(axes.ravel(), names):
        arr = all_arrays[name]
        p = arr[arr > 0]
        if p.size == 0:
            ax.set_title(f"{_disp(name)} (empty)")
            continue
        lo, hi = p.min(), p.max()
        bins = np.logspace(np.log10(lo), np.log10(hi if hi > lo else lo * 10), 100)

        w = np.full(p.shape, 100.0 / max(arr.size, 1))
        ax.hist(p, bins=bins, weights=w, histtype="step", lw=1.5, color="tab:green",
                label=f"All (n={arr.size}, med {np.median(p):.2e})")
        ax.axvline(np.median(p), color="tab:green", ls="--", lw=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("% of All")
        ax.set_title(_disp(name))
        ax.legend(fontsize=8)
    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"viewspace_stats_all_{iteration}.png"), dpi=120)
    plt.close(fig)

    # ---- 시각화 3: pc._gaussian_scene_num 값 분포 (scene0 / scene1 / 전체 visible) ----
    if scene_vals:
        _plot_scene_num(
            plt, scene_vals, colors, scene_key,
            fig_title,
            os.path.join(out_dir, f"viewspace_scene_num_{iteration}.png"),
        )
