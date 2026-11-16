"""Track per-iteration gradient / Adam / size statistics of a sampled set of Gaussians.

Usage (train.py):
    tracker = GaussianTracker(gaussians, out_dir, num_track=1000)   # after training_setup / restore
    ...
    loss.backward()
    tracker.record(iteration, gaussians, render_pkg)                # before densification / optimizer.step
    ...
    tracker.finalize(iteration, gaussians)                          # at the last iteration

GaussianModel must call the two hooks (see gaussian_model.py):
    densification_postfix(..., parent_idx=...) -> tracker.on_append(num_new, parent_idx)
    prune_points(mask)                         -> tracker.on_prune(valid_points_mask)

Storage layout (out_dir):
    chunk_XXXXXX.npz   rows recorded since the previous flush
        iters   (T,)      recorded iterations (by default ONE chunk holds the whole run)
        counts  (T,)      number of rows of each iteration (rows are stored iteration by iteration)
        track_id (R,)     track id of each row
        values  (R, 16)   float32, columns = COLUMNS
    track_meta.npz     written by finalize(): lineage, deaths, final gaussian_scene_num

Every tracked Gaussian has a track id. A child created by clone / split / random split gets a
NEW track id and a parent pointer, so "the child inherits the parent's values" is stored as a
lineage instead of physically duplicating the history. load_history() stitches the inherited
part (parent rows up to the child's birth iteration) back together.
"""
import glob
import os

import numpy as np
import torch

COLUMNS = [
    "grad_x", "grad_y", "grad_z",                 # d loss / d xyz from this iteration's backward
    "adam_m_x", "adam_m_y", "adam_m_z",           # Adam exp_avg of xyz        (state BEFORE this iteration's step)
    "adam_sqrt_v_x", "adam_sqrt_v_y", "adam_sqrt_v_z",  # sqrt(Adam exp_avg_sq) of xyz (same timing)
    "scale_0", "scale_1", "scale_2",              # activated scaling (world units)
    "vs_abs",                                     # |viewspace_abs.grad[:, :2]|
    "vs_points",                                  # |viewspace_points.grad[:, :2]|
    "ratio",                                      # vs_abs / vs_points (nan if vs_points == 0)
    "visible",                                    # 1 if radii > 0 in this iteration's view
]


class GaussianTracker:
    def __init__(self, gaussians, out_dir, num_track=1000, record_every=1, only_visible=False,
                 max_tracked=None, inherit_prob=1.0, random_split_max_children=2,
                 flush_every=None, max_buffer_mb=4096):
        """
        num_track    : number of Gaussians sampled at start
        record_every : record every k-th iteration (1 = every iteration)
        only_visible : if True, only store rows of Gaussians visible in this iteration's view
        max_tracked  : cap on simultaneously tracked Gaussians; once reached, newly created
                       children are no longer tracked (None = no cap)
        inherit_prob : probability that a child of a tracked Gaussian (clone / split /
                       random split) inherits the track; decided independently per child
        random_split_max_children : for random_split_explore, at most this many children of
                       one parent are candidates for inheriting (chosen at random, then
                       inherit_prob is applied to each). None = all children are candidates
        flush_every  : write a chunk file every this many iterations
                       (None = keep everything in RAM and write once in finalize())
        max_buffer_mb: safety valve; a chunk is written early whenever the rows waiting
                       in RAM exceed this size (None = no limit)
        """
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        stale = glob.glob(os.path.join(out_dir, "chunk_*.npz"))
        if stale:
            print(f"[Tracker] removing {len(stale)} chunk files of a previous run in {out_dir}")
            for f in stale:
                os.remove(f)

        self.record_every = int(record_every)
        self.only_visible = bool(only_visible)
        self.max_tracked = max_tracked
        self.inherit_prob = float(inherit_prob)
        self.random_split_max_children = random_split_max_children
        self.flush_every = int(flush_every) if flush_every else None
        self.max_buffer_mb = max_buffer_mb
        self._buffer_bytes = 0
        self.iteration = 0

        n = gaussians._xyz.shape[0]
        device = gaussians._xyz.device
        k = min(int(num_track), n)
        init_index = torch.randperm(n, device=device)[:k].sort().values
        self.track_id = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.track_id[init_index] = torch.arange(k, dtype=torch.int64, device=device)
        self.next_id = k
        self.n_alive = k
        self.init_index = init_index.cpu().numpy()

        # lineage / deaths (lists of numpy arrays)
        self._child, self._parent, self._birth = [], [], []
        self._dead, self._dead_iter = [], []
        # rows waiting to be flushed
        self._iters, self._counts, self._ids, self._vals = [], [], [], []
        self._rows_total = 0

        gaussians.tracker = self
        print(f"[Tracker] tracking {k} of {n} Gaussians -> {out_dir}")

    # ------------------------------------------------------------------ hooks
    @torch.no_grad()
    def on_append(self, num_new, parent_idx=None, max_children=None):
        """Called from densification_postfix BEFORE any pruning.
        parent_idx: (num_new,) index of each new Gaussian's parent in the pre-append ordering,
                    or None if the new Gaussians have no parent (e.g. seeds).
        max_children: at most this many children per parent may inherit (None = no limit)."""
        device = self.track_id.device
        new_ids = torch.full((num_new,), -1, dtype=torch.int64, device=device)
        if parent_idx is not None and num_new > 0:
            pid = self.track_id[parent_idx]
            sel = pid >= 0
            if max_children is not None and bool(sel.any()):
                # keep a random subset of at most `max_children` children per parent
                cand = sel.nonzero(as_tuple=True)[0]
                par = parent_idx[cand]
                order = torch.argsort(torch.rand(cand.numel(), device=device))
                order = order[torch.argsort(par[order], stable=True)]     # grouped by parent, random inside
                _, counts = torch.unique_consecutive(par[order], return_counts=True)
                starts = torch.cumsum(counts, 0) - counts
                rank = torch.arange(cand.numel(), device=device) - torch.repeat_interleave(starts, counts)
                sel[cand[order[rank >= int(max_children)]]] = False
            if self.inherit_prob < 1.0:            # each child inherits with this probability
                sel &= torch.rand(num_new, device=device) < self.inherit_prob
            k = int(sel.sum())
            if k > 0 and self.max_tracked is not None:
                room = max(int(self.max_tracked) - self.n_alive, 0)
                if k > room:                       # drop a random subset of the children
                    cand = sel.nonzero(as_tuple=True)[0]
                    drop = cand[torch.randperm(k, device=device)[room:]]
                    sel[drop] = False
                    k = room
            if k > 0:
                ids = torch.arange(self.next_id, self.next_id + k, dtype=torch.int64, device=device)
                new_ids[sel] = ids
                self._child.append(ids.cpu().numpy())
                self._parent.append(pid[sel].cpu().numpy())
                self._birth.append(np.full(k, self.iteration, dtype=np.int32))
                self.next_id += k
                self.n_alive += k
        self.track_id = torch.cat((self.track_id, new_ids))

    @torch.no_grad()
    def on_prune(self, valid_points_mask):
        """Called from prune_points with the mask of Gaussians that are KEPT."""
        dead = self.track_id[~valid_points_mask]
        dead = dead[dead >= 0]
        if dead.numel() > 0:
            self._dead.append(dead.cpu().numpy())
            self._dead_iter.append(np.full(dead.numel(), self.iteration, dtype=np.int32))
            self.n_alive -= int(dead.numel())
        self.track_id = self.track_id[valid_points_mask]

    # ------------------------------------------------------------------ record
    @staticmethod
    def _xyz_adam_state(gaussians):
        for group in gaussians.optimizer.param_groups:
            if group["name"] == "xyz":
                return gaussians.optimizer.state.get(group["params"][0], None)
        return None

    @staticmethod
    def _vs_norm(t, idx):
        g = getattr(t, "grad", None)
        if g is None:
            return torch.full((idx.numel(),), float("nan"), device=idx.device)
        return torch.linalg.vector_norm(g[idx, :2], dim=-1)

    @torch.no_grad()
    def record(self, iteration, gaussians, render_pkg):
        """Call right after loss.backward(), before densification and optimizer.step()."""
        self.iteration = int(iteration)
        n = gaussians._xyz.shape[0]
        if self.track_id.shape[0] != n:
            raise RuntimeError(f"[Tracker] track_id has {self.track_id.shape[0]} entries but the model "
                               f"has {n} Gaussians: a prune/append hook is missing in GaussianModel")

        if iteration % self.record_every == 0:
            radii = render_pkg["radii"]
            idx = (self.track_id >= 0).nonzero(as_tuple=True)[0]
            if self.only_visible:
                idx = idx[radii[idx] > 0]
            if idx.numel() > 0:
                k = idx.numel()
                device = idx.device
                grad = gaussians._xyz.grad
                g = grad[idx] if grad is not None else torch.zeros((k, 3), device=device)
                st = self._xyz_adam_state(gaussians)
                if st is not None and "exp_avg" in st:
                    m = st["exp_avg"][idx]
                    sqrt_v = st["exp_avg_sq"][idx].sqrt()
                else:                                   # before the first optimizer step
                    m = torch.zeros((k, 3), device=device)
                    sqrt_v = torch.zeros((k, 3), device=device)
                scale = gaussians.get_scaling[idx]
                a = self._vs_norm(render_pkg["viewspace_abs"], idx)
                p = self._vs_norm(render_pkg["viewspace_points"], idx)
                ratio = torch.where(p > 0, a / p.clamp_min(1e-30), torch.full_like(p, float("nan")))
                vis = (radii[idx] > 0).float()
                vals = torch.cat((g, m, sqrt_v, scale, a[:, None], p[:, None],
                                  ratio[:, None], vis[:, None]), dim=1)
                self._iters.append(self.iteration)
                self._counts.append(k)
                self._ids.append(self.track_id[idx].to(torch.int32).cpu().numpy())
                self._vals.append(vals.float().cpu().numpy())
                self._buffer_bytes += self._vals[-1].nbytes + self._ids[-1].nbytes

        if self.flush_every is not None and iteration % self.flush_every == 0:
            self.flush()
        elif self.max_buffer_mb is not None and self._buffer_bytes > self.max_buffer_mb * 2**20:
            self.flush()

    def flush(self):
        if not self._iters:
            return
        vals = np.concatenate(self._vals)
        path = os.path.join(self.out_dir, f"chunk_{self._iters[0]:06d}.npz")
        np.savez(path,
                 iters=np.asarray(self._iters, dtype=np.int32),
                 counts=np.asarray(self._counts, dtype=np.int32),
                 track_id=np.concatenate(self._ids),
                 values=vals,
                 columns=np.array(COLUMNS))
        self._rows_total += vals.shape[0]
        print(f"[Tracker] iter {self._iters[0]}-{self._iters[-1]}: {vals.shape[0]} rows "
              f"({vals.nbytes / 2**20:.1f} MB), alive tracked={self.n_alive}, total rows={self._rows_total}")
        self._iters, self._counts, self._ids, self._vals = [], [], [], []
        self._buffer_bytes = 0

    # ------------------------------------------------------------------ finalize
    @torch.no_grad()
    def finalize(self, iteration, gaussians, prune_dead=True):
        """Flush, save lineage + final gaussian_scene_num, and (prune_dead) delete the rows of
        tracks that were pruned and have no surviving descendant."""
        self.flush()
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
        child, parent, birth = cat(self._child, np.int64), cat(self._parent, np.int64), cat(self._birth, np.int32)

        idx = (self.track_id >= 0).nonzero(as_tuple=True)[0]
        final_ids = self.track_id[idx].cpu().numpy()
        scene_num = gaussians._gaussian_scene_num.detach().reshape(-1).float()

        np.savez(os.path.join(self.out_dir, "track_meta.npz"),
                 columns=np.array(COLUMNS),
                 num_tracks=self.next_id,
                 init_track_id=np.arange(len(self.init_index)),
                 init_gaussian_index=self.init_index,
                 child=child, parent=parent, birth_iter=birth,            # lineage
                 dead_track_id=cat(self._dead, np.int64), dead_iter=cat(self._dead_iter, np.int32),
                 final_iteration=int(iteration),
                 final_track_id=final_ids,                                # alive at the end
                 final_gaussian_index=idx.cpu().numpy(),
                 final_scene_num=scene_num[idx].cpu().numpy(),            # gaussian_scene_num of tracked
                 final_scene_num_all=scene_num.cpu().numpy())             # gaussian_scene_num of all
        print(f"[Tracker] finalized at iter {iteration}: {len(final_ids)} tracked Gaussians alive "
              f"({self.next_id} tracks created in total)")

        if prune_dead:
            keep = _keep_ids(self.next_id, child, parent, final_ids)
            removed = _rewrite_chunks(self.out_dir, keep)
            print(f"[Tracker] deleted {removed} rows of pruned tracks without surviving descendants")


# ---------------------------------------------------------------------- numpy helpers
def _parent_array(num_tracks, child, parent):
    par = np.full(int(num_tracks), -1, dtype=np.int64)
    par[child] = parent
    return par


def _keep_ids(num_tracks, child, parent, alive_ids):
    """Bool mask over track ids: alive tracks and all of their ancestors."""
    par = _parent_array(num_tracks, child, parent)
    keep = np.zeros(int(num_tracks), dtype=bool)
    frontier = np.unique(np.asarray(alive_ids, dtype=np.int64))
    while frontier.size:
        keep[frontier] = True
        frontier = par[frontier]
        frontier = np.unique(frontier[frontier >= 0])
        frontier = frontier[~keep[frontier]]
    return keep


def _rewrite_chunks(out_dir, keep):
    removed = 0
    for path in sorted(glob.glob(os.path.join(out_dir, "chunk_*.npz"))):
        with np.load(path) as ch:
            ids, vals, cols = ch["track_id"], ch["values"], ch["columns"]
            row_iter = np.repeat(ch["iters"], ch["counts"])
        m = keep[ids]
        removed += int((~m).sum())
        if m.all():
            continue
        iters, counts = np.unique(row_iter[m], return_counts=True)
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, iters=iters.astype(np.int32), counts=counts.astype(np.int32),
                 track_id=ids[m], values=vals[m], columns=cols)
        os.replace(tmp, path)
    return removed


# ---------------------------------------------------------------------- loading
def load_meta(out_dir):
    with np.load(os.path.join(out_dir, "track_meta.npz")) as m:
        return {k: m[k] for k in m.files}


def load_history(out_dir, track_id, inherit=True, meta=None):
    """History of one track as a dict: 'iters', 'source_track_id' and one array per COLUMNS entry.

    inherit=True prepends the values inherited from its ancestors (each ancestor's rows up to
    the iteration at which the next descendant was created)."""
    meta = load_meta(out_dir) if meta is None else meta
    par = _parent_array(meta["num_tracks"], meta["child"], meta["parent"])
    birth = np.zeros(int(meta["num_tracks"]), dtype=np.int64)
    birth[meta["child"]] = meta["birth_iter"]

    segments, cur, upper = [], int(track_id), np.iinfo(np.int64).max
    while cur >= 0:
        segments.append((cur, upper))            # rows of `cur` with iteration <= upper
        if not inherit:
            break
        upper, cur = min(upper, int(birth[cur])), int(par[cur])

    its, src, vals = [], [], []
    for path in sorted(glob.glob(os.path.join(out_dir, "chunk_*.npz"))):
        with np.load(path) as ch:
            ids = ch["track_id"]
            row_iter = np.repeat(ch["iters"], ch["counts"])
            m = np.zeros(ids.shape[0], dtype=bool)
            for sid, up in segments:
                m |= (ids == sid) & (row_iter <= up)
            if m.any():
                its.append(row_iter[m]); src.append(ids[m]); vals.append(ch["values"][m])
    if not its:
        return {"iters": np.zeros(0, dtype=np.int32), "source_track_id": np.zeros(0, dtype=np.int32),
                **{c: np.zeros(0, dtype=np.float32) for c in COLUMNS}}
    its, src, vals = np.concatenate(its), np.concatenate(src), np.concatenate(vals)
    order = np.argsort(its, kind="stable")
    out = {"iters": its[order], "source_track_id": src[order]}
    out.update({c: vals[order, i] for i, c in enumerate(COLUMNS)})
    return out
