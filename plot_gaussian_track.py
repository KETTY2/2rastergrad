"""학습 종료 시점에 살아남은 추적 Gaussian 중 gaussian_scene_num 이 가장 작은 / 큰 것들을 골라
iteration 별 기록을 그린다.  (gaussian_tracker.py 가 저장한 chunk_*.npz + track_meta.npz 사용)

    python plot_gaussian_track.py <model_path>/gaussian_track            # 작은 2개 + 큰 2개
    python plot_gaussian_track.py <model_path>/gaussian_track --num 3
    python plot_gaussian_track.py <model_path>/gaussian_track --min_scene_num 10   # 기본 5 이상에서만 선택
    python plot_gaussian_track.py <model_path>/gaussian_track --track_ids 12 857

행(row):
    |grad|         = sqrt(grad_x^2 + grad_y^2 + grad_z^2)
    |m / sqrt(v)|  = (adam_m_x/adam_sqrt_v_x, adam_m_y/adam_sqrt_v_y, adam_m_z/adam_sqrt_v_z) 의 크기
    scale_0,1,2
    ratio          = vs_abs / vs_points
열(column): 선택된 Gaussian 하나씩 (파랑 = scene_num 작은 쪽, 빨강 = 큰 쪽)

자식 Gaussian 은 태어나기 전 구간을 부모(조상)의 기록으로 이어 붙인다 (회색 점선 = split/clone 시점).
torch 없이 numpy + matplotlib 만 사용한다.
"""
import argparse
import glob
import os

import numpy as np


def lineage_segments(track_id, par, birth, inherit=True):
    """[(track id, 그 track 의 기록을 쓸 마지막 iteration)], 자기 자신 -> 조상 순."""
    segs, cur, upper = [], int(track_id), np.iinfo(np.int64).max
    while cur >= 0:
        segs.append((cur, upper))
        if not inherit:
            break
        upper, cur = min(upper, int(birth[cur])), int(par[cur])
    return segs


def load_tracks(track_dir, segments_by_track):
    """chunk 파일을 한 번씩만 읽어서 여러 track 의 history 를 동시에 뽑는다."""
    needed = np.unique([sid for segs in segments_by_track.values() for sid, _ in segs])
    paths = sorted(p for p in glob.glob(os.path.join(track_dir, "chunk_*.npz"))
                   if not p.endswith(".tmp.npz"))
    if not paths:
        raise FileNotFoundError(f"no chunk_*.npz in {track_dir}")
    its, ids, vals, columns = [], [], [], None
    for path in paths:
        with np.load(path) as ch:
            columns = [str(c) for c in ch["columns"]]
            cid = ch["track_id"]
            m = np.isin(cid, needed)
            if not m.any():
                continue
            its.append(np.repeat(ch["iters"], ch["counts"])[m])
            ids.append(cid[m])
            vals.append(ch["values"][m])
        print(f"  read {os.path.basename(path)}: {int(m.sum())} rows")
    its, ids, vals = np.concatenate(its), np.concatenate(ids), np.concatenate(vals)

    out = {}
    for tid, segs in segments_by_track.items():
        m = np.zeros(ids.shape[0], dtype=bool)
        for sid, up in segs:
            m |= (ids == sid) & (its <= up)
        order = np.argsort(its[m], kind="stable")
        h = {"iters": its[m][order], "source_track_id": ids[m][order]}
        h.update({c: vals[m][order, i].astype(np.float64) for i, c in enumerate(columns)})
        out[tid] = h
    return out


def derived(h):
    """그릴 값 계산."""
    grad = np.sqrt(h["grad_x"] ** 2 + h["grad_y"] ** 2 + h["grad_z"] ** 2)
    comps = []
    for a in "xyz":
        m, sv = h[f"adam_m_{a}"], h[f"adam_sqrt_v_{a}"]
        comps.append(np.divide(m, sv, out=np.zeros_like(m), where=sv > 0))   # v == 0 이면 0
    adam = np.sqrt(sum(c ** 2 for c in comps))
    return grad, adam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track_dir", help="chunk_*.npz 와 track_meta.npz 가 있는 폴더 (<model_path>/gaussian_track)")
    ap.add_argument("--num", type=int, default=2, help="작은 쪽 / 큰 쪽에서 각각 뽑을 개수")
    ap.add_argument("--min_scene_num", type=float, default=5.0,
                    help="gaussian_scene_num 이 이 값 이상인 Gaussian 중에서만 뽑는다 (0 = 제한 없음)")
    ap.add_argument("--track_ids", type=int, nargs="+", default=None, help="직접 지정 (지정하면 --num 무시)")
    ap.add_argument("--no_inherit", action="store_true", help="조상 기록을 이어 붙이지 않음")
    ap.add_argument("--out", default=None, help="저장할 png 경로")
    args = ap.parse_args()

    with np.load(os.path.join(args.track_dir, "track_meta.npz")) as f:
        meta = {k: f[k] for k in f.files}
    num_tracks = int(meta["num_tracks"])
    final_iter = int(meta["final_iteration"])
    final_ids = meta["final_track_id"].astype(np.int64)
    final_sn = meta["final_scene_num"].astype(np.float64)
    par = np.full(num_tracks, -1, dtype=np.int64)
    par[meta["child"]] = meta["parent"]
    birth = np.zeros(num_tracks, dtype=np.int64)
    birth[meta["child"]] = meta["birth_iter"]
    scene_num_of = dict(zip(final_ids.tolist(), final_sn.tolist()))

    print(f"alive tracked Gaussians: {final_ids.size}, gaussian_scene_num "
          f"min={final_sn.min():.3g} med={np.median(final_sn):.3g} max={final_sn.max():.3g}")

    # ---- 선택: scene_num 오름차순으로 앞 num 개 (작은 쪽) / 뒤 num 개 (큰 쪽)
    if args.track_ids:
        missing = [t for t in args.track_ids if t not in scene_num_of]
        if missing:
            raise SystemExit(f"track ids not alive at the end: {missing}")
        picks = [(t, "tab:gray", "picked") for t in args.track_ids]
    else:
        # gaussian_scene_num >= min_scene_num 인 것만 후보로 둔다
        cand = np.nonzero(final_sn >= args.min_scene_num)[0]
        print(f"candidates with gaussian_scene_num >= {args.min_scene_num:g}: {cand.size} of {final_ids.size}")
        if cand.size < 2:
            raise SystemExit("not enough candidates; lower --min_scene_num")
        k = min(args.num, cand.size // 2)
        order = cand[np.argsort(final_sn[cand], kind="stable")]
        picks = [(int(final_ids[i]), "tab:blue", "low") for i in order[:k]] + \
                [(int(final_ids[i]), "tab:red", "high") for i in order[::-1][:k][::-1]]

    segs = {t: lineage_segments(t, par, birth, inherit=not args.no_inherit) for t, _, _ in picks}
    hist = load_tracks(args.track_dir, segs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ["|grad|  (grad_x, grad_y, grad_z)", "|m / sqrt(v)|  (Adam, xyz)", "scale", "ratio  (vs_abs / vs_points)"]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(5.2 * len(picks), 3.0 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    for j, (tid, color, kind) in enumerate(picks):
        h = hist[tid]
        it = h["iters"]
        grad, adam = derived(h)
        births = sorted(up for _, up in segs[tid] if up <= final_iter)
        print(f"  [{kind}] track {tid}: gaussian_scene_num={scene_num_of[tid]:.4g}, rows={it.size}, "
              f"iters {it.min() if it.size else '-'}..{it.max() if it.size else '-'}, "
              f"ancestors={len(segs[tid]) - 1}")

        pos = lambda v: np.where(v > 0, v, np.nan)          # log 축: 0 / nan 은 그리지 않음
        axes[0, j].plot(it, pos(grad), ".", ms=1.5, color=color, rasterized=True)
        axes[1, j].plot(it, adam, "-", lw=0.5, color=color, rasterized=True)
        for s, c in zip(("scale_0", "scale_1", "scale_2"), ("tab:green", "tab:orange", "tab:purple")):
            axes[2, j].plot(it, h[s], "-", lw=1.0, color=c, label=s)
        axes[3, j].plot(it, pos(h["ratio"]), ".", ms=1.5, color=color, rasterized=True)
        axes[3, j].axhline(1.0, color="k", lw=0.6, ls=":")

        for i in (0, 2, 3):
            axes[i, j].set_yscale("log")
        for i in range(len(rows)):
            for b in births:                                 # split / clone 으로 태어난 시점
                axes[i, j].axvline(b, color="gray", lw=0.6, ls="--", alpha=0.6)
            axes[i, j].grid(alpha=0.25)
            axes[i, j].set_xlim(0, final_iter)
        axes[0, j].set_title(f"{kind}: track {tid}\ngaussian_scene_num = {scene_num_of[tid]:.4g}",
                             color=color, fontsize=10)
        axes[2, j].legend(fontsize=7, loc="upper right")
        axes[-1, j].set_xlabel("iteration")
    for i, name in enumerate(rows):
        axes[i, 0].set_ylabel(name, fontsize=9)
    fig.suptitle("Tracked Gaussians with the lowest / highest gaussian_scene_num "
                 "(dashed gray = split / clone, earlier part inherited from the parent)", fontsize=11)
    fig.tight_layout()
    out = args.out or os.path.join(args.track_dir, "track_scene_num_low_high.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
