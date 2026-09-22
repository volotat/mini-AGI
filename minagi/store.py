#!/usr/bin/env python3
"""
The weights directory IS the model.

Everything the model works with is a separate file. There is no monolithic
checkpoint; this directory is what training reads at the start of a session,
what it writes at every evaluation, and what inference loads from.

    weights/
      manifest.json          what exists, its shape, and where it came from
      core.npz               embeddings, attention, norms, halting head
      routers.npz            one entry per call site - which experts it reaches
      optim.npz              Adam moments for the trunk and the routers
      experts/               one file per expert: w1, w3, w2 and that
        e00000.npz           expert's own Adam moments
        e00001.npz
        ...

Only the experts are split out, because an expert is the unit that gets paged,
grown and pruned - a new expert really is a new file. Everything else is always
resident together and gains nothing from being separate, so it is bundled.

When the model grows an expert, a new file appears in experts/. When it prunes
one, that file is deleted. `ls weights/experts | wc -l` is the parameter count
in the most literal sense available.

Writes are atomic: every file is written to a .tmp and renamed, so an
interrupted save cannot leave a half-written weight behind.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

from .precision import pack_bf16, unpack_bf16

CORE, ROUTERS, EXPERTS, OPTIM = "core", "routers", "experts", "optim"


class WeightShadow:
    """Temporary storage for a dry read's model or changed experts."""

    def __init__(self, source):
        source = os.path.abspath(source)
        parent = os.path.dirname(source)
        while not os.path.isdir(parent):
            above = os.path.dirname(parent)
            if above == parent:
                parent = None
                break
            parent = above
        try:
            self._tmp = tempfile.TemporaryDirectory(
                prefix=".minagi-dry-", dir=parent,
                ignore_cleanup_errors=True)
        except OSError:
            self._tmp = tempfile.TemporaryDirectory(
                prefix="minagi-dry-", ignore_cleanup_errors=True)
        self.path = os.path.join(self._tmp.name, "weights")

    def cleanup(self):
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()


def shadow(path):
    """Return temporary storage for a dry read of a weights directory."""
    return WeightShadow(path)


def _is_expert(key):
    return ".experts." in key and key.startswith("pool.")


def _is_router(key):
    return key.endswith("router.weight") or key.endswith("depth_emb")


def expert_index(key):
    """pool.experts.12.w1.weight -> 12"""
    parts = key.split(".")
    return int(parts[parts.index("experts") + 1])


def expert_leaf(key):
    """
    What to call this tensor inside the expert's own file.

        pool.experts.12.w1.weight           -> w1
        pool.experts.12.blocks.0.w1.weight  -> b0_w1

    Taking the second-to-last name instead would call both of those `w1`, and
    a deeper expert would quietly save only its first block.
    """
    parts = key.split(".")
    i = parts.index("experts") + 2          # past the index
    tail = parts[i:-1]                      # drop the trailing "weight"
    if tail[0] == "blocks":
        return f"b{tail[1]}_{tail[2]}"
    return tail[0]


def save(model, path, step=None, val=None, opt=None, cfg=None, verbose=False,
         extra=None):
    """Write the whole model out: bundles for the resident parts, one file per expert."""
    os.makedirs(os.path.join(path, EXPERTS), exist_ok=True)

    # A paged pool keeps its experts as files already - it has no
    # pool.experts.N tensors to gather, only the VRAM slots. Grouping its
    # state_dict by the usual pattern finds nothing, writes an empty expert
    # list, and then deletes every file the manifest does not mention, which
    # is all of them. So a paged pool is asked to flush itself instead.
    pool = getattr(model, "pool", None)
    if pool is not None and hasattr(pool, "tiers"):
        return _save_paged(model, pool, path, step, val, opt, cfg, verbose,
                           extra)

    sd = model.state_dict()

    by_expert, core, routers = {}, {}, {}
    for k, v in sd.items():
        if _is_expert(k):
            by_expert.setdefault(expert_index(k), {})[expert_leaf(k)] = v
        elif _is_router(k):
            routers[k] = v.detach().cpu().numpy()
        else:
            core[k] = v.detach().cpu().numpy()

    total = _savez(os.path.join(path, "core.npz"), core)
    total += _savez(os.path.join(path, "routers.npz"), routers)

    entries = []
    gate = sd.get("pool.gate")
    moments = _expert_moments(opt, model) if opt is not None else {}
    for i in sorted(by_expert):
        parts = by_expert[i]
        arrays = {k: v.detach().cpu().numpy() for k, v in parts.items()}
        # an expert's Adam moments belong to the expert, so they live in the
        # expert's own file. A file that holds only weights cannot be paged
        # back in and resumed - it comes back with no history, or worse, with
        # whatever history the slot it lands in happened to have.
        arrays.update(moments.get(i, {}))
        n = _savez(os.path.join(path, EXPERTS, "e%05d.npz" % i), arrays)
        total += n
        params = sum(a.size for k, a in arrays.items() if not k.endswith(("_m", "_v")))
        entries.append({"id": i, "file": "e%05d.npz" % i,
                        "params": int(params), "bytes": int(n),
                        "moments": any(k.endswith("_m") for k in arrays),
                        "gate": float(gate[i]) if gate is not None
                        and i < gate.numel() else 1.0})
    live = {e["file"] for e in entries}
    ed = os.path.join(path, EXPERTS)
    removed = 0
    for f in os.listdir(ed):
        if f.endswith((".npy", ".npz")) and f not in live:
            os.remove(os.path.join(ed, f))
            removed += 1

    if opt is not None:
        total += _save_optim(opt, model, path)

    d_ff = d_model = None
    for i in sorted(by_expert)[:1]:
        d_ff, d_model = by_expert[i]["w1"].shape
    manifest = {
        "step": step, "val": val, "cfg": cfg,
        "n_experts": len(entries), "d_model": d_model, "d_ff": d_ff,
        "core_tensors": sorted(core), "router_tensors": sorted(routers),
        "experts": entries, "total_bytes": int(total),
        "removed_expert_files": removed,
    }
    tmp = os.path.join(path, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(path, "manifest.json"))
    if verbose:
        n_files = len(entries) + 3
        print(f"  weights/: {len(entries)} experts + 3 bundles = {n_files} files, "
              f"{total/1e6:.1f} MB"
              + (f", {removed} expert files removed" if removed else ""))
    return len(entries), total


def _expert_moments(opt, model):
    """Adam's two moments per expert, keyed by expert id."""
    name_of = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in opt.param_groups:
        for p in group["params"]:
            n = name_of.get(id(p))
            if n is None or not _is_expert(n):
                continue
            st = opt.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            i, leaf = expert_index(n), expert_leaf(n)
            out.setdefault(i, {})[leaf + "_m"] = st["exp_avg"].cpu().numpy()
            out[i][leaf + "_v"] = st["exp_avg_sq"].cpu().numpy()
    return out


def _save_paged(model, pool, path, step, val, opt, cfg, verbose, extra=None):
    """
    Save a model whose experts are already on disk.

    The expert files are the pool, so saving means parking whatever is
    resident - with its optimiser moments - and writing the manifest around
    what is there. Nothing is gathered and nothing is deleted.
    """
    if os.path.abspath(pool.tiers.write_path) != os.path.abspath(pool.tiers.path):
        raise RuntimeError(
            "cannot checkpoint a pool with a temporary expert overlay")
    pool.flush()
    core, routers = {}, {}
    for k, v in model.state_dict().items():
        if k.startswith("pool."):
            if k.endswith(("gate", "segment_router.weight")):
                routers[k] = v.detach().cpu().numpy()
            continue                      # w1/w3/w2 are slots, not experts
        (routers if _is_router(k) else core)[k] = v.detach().cpu().numpy()
    total = _savez(os.path.join(path, "core.npz"), core)
    total += _savez(os.path.join(path, "routers.npz"), routers)
    if opt is not None:
        total += _save_optim(opt, model, path)
    ed = os.path.join(path, EXPERTS)
    entries = []
    gate = pool.gate.detach().cpu()
    # An expert's FILE is named by its uid, not by its position. Pruning does
    # not renumber the directory, so the two diverge as soon as anything is
    # deleted, and building the name from the position would skip every expert
    # whose file no longer matches its index.
    uid = getattr(pool, "uid", None)
    missing = []
    for i in range(pool.n_experts()):
        u = int(uid[i]) if uid is not None and i < uid.numel() else i
        f = "e%05d.npz" % u
        p = os.path.join(ed, f)
        if not os.path.exists(p):
            missing.append(u)
            continue
        n = os.path.getsize(p)
        total += n
        entries.append({"id": u, "file": f, "bytes": int(n),
                        "params": pool.d_ff * pool.d_model * 3,
                        "moments": True,
                        "gate": float(gate[i]) if i < gate.numel() else 1.0})
    if missing:
        print(f"  WARNING: {len(missing)} experts have no file on disk "
              f"({missing[:6]}...) - the checkpoint will name fewer experts "
              f"than the pool holds", flush=True)
    d = dict(cfg or {})
    d["pool_resident"] = pool.resident
    # Record the ceiling as at least what the pool actually holds. Growth is
    # not capped in the read path, so a stale `pool_max` from an older config
    # leaves the routers a row short of the experts on disk and the directory
    # cannot be loaded back without widening it by hand.
    d["pool_max"] = max(int(d.get("pool_max") or 0), len(entries))
    # Which experts have ever been resident, and when each was admitted.
    # This belongs in the checkpoint: the cold-start sweep exists to give an
    # expert that has never been on the card its one chance, and if the flags
    # reset every run the sweep restarts every run and the working set is
    # never allowed to settle.
    if hasattr(pool, "telemetry"):
        # Everything the pool knows about its own experts. Rebuilt from
        # nothing every run otherwise, so a long run would end able to say
        # what the model is and nothing about how it got there.
        tel = pool.telemetry()
        manifest_tel = tel
        d["pool_ever"] = tel["ever"]
        d["pool_since"] = tel["since"]
    else:
        manifest_tel = None
    manifest = {"step": step, "val": val, "cfg": d,
                "telemetry": manifest_tel,
                # The pool's own count, NOT len(entries). The router
                # tensors were just written with one row per expert, and
                # loading sizes them from this number - so a file count that
                # disagrees produces a shape mismatch rather than a warning.
                "n_experts": int(pool.n_experts()), "d_model": pool.d_model,
                "d_ff": pool.d_ff, "paged": True,
                "core_tensors": sorted(core), "router_tensors": sorted(routers),
                "experts": entries, "total_bytes": int(total)}
    manifest.update(extra or {})       # where the reader had got to
    tmp = os.path.join(path, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(path, "manifest.json"))
    if verbose:
        print(f"  weights/: {len(entries)} experts + 3 bundles, "
              f"{total/1e6:.1f} MB (paged - experts were already on disk)")
    return len(entries), total


def _savez(path, arrays):
    """Atomic bundle write."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return sum(a.nbytes for a in arrays.values())


def _save_optim(opt, model, path):
    """
    Adam moments for everything that is not an expert.

    Expert moments live in the expert's own file, next to its weights, because
    that is where they belong - an expert is not fully described by its weights
    alone. This bundle holds the rest: the trunk, the routers, the gates.
    """
    name_of = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st or "exp_avg" not in st:
                continue
            n = name_of.get(id(p))
            if n is None or _is_expert(n):
                continue
            # bf16: two thirds of this file is moments, and narrowing them
            # costs a thousandth of their magnitude. See minagi/precision.py.
            out[n + "|m"] = pack_bf16(st["exp_avg"].detach().cpu())
            out[n + "|v"] = pack_bf16(st["exp_avg_sq"].detach().cpu())
            if st.get("step") is not None:
                out[n + "|t"] = np.asarray(float(st["step"]))
    return _savez(os.path.join(path, "optim.npz"), out) if out else 0


def load(model, path, opt=None, device=None, strict=False, verbose=False):
    """Rebuild a model's tensors from the directory."""
    device = device or next(model.parameters()).device
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    sd = {}
    core = np.load(os.path.join(path, "core.npz"))
    routers = np.load(os.path.join(path, "routers.npz"))
    for k in core.files:
        sd[k] = torch.from_numpy(core[k])
    for k in routers.files:
        sd[k] = torch.from_numpy(routers[k])
    d_model, d_ff = man["d_model"], man["d_ff"]
    n1 = d_ff * d_model
    for e in man["experts"]:
        f = os.path.join(path, EXPERTS, e["file"])
        i = e["id"]
        if f.endswith(".npz"):
            z = np.load(f)
            for leaf in z.files:
                if leaf.endswith(("_m", "_v")):
                    continue                       # optimiser state, not weights
                if leaf.startswith("b") and "_" in leaf:
                    b, nm = leaf[1:].split("_", 1)
                    key = f"pool.experts.{i}.blocks.{b}.{nm}.weight"
                else:
                    key = f"pool.experts.{i}.{leaf}.weight"
                sd[key] = torch.from_numpy(z[leaf])
        else:
            # the older flat layout, before moments moved into the file
            flat = torch.from_numpy(np.load(f))
            sd[f"pool.experts.{i}.w1.weight"] = flat[:n1].view(d_ff, d_model)
            sd[f"pool.experts.{i}.w3.weight"] = flat[n1:2 * n1].view(d_ff, d_model)
            sd[f"pool.experts.{i}.w2.weight"] = flat[2 * n1:].view(d_model, d_ff)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if opt is not None:
        _load_optim(opt, model, path)
    if verbose:
        print(f"  loaded weights/ at step {man.get('step')}: "
              f"{man['n_experts']} experts, {man['total_bytes']/1e6:.1f} MB"
              + (f", {len(missing)} tensors not in the directory" if missing else ""))
    return man, missing, unexpected


def _load_optim(opt, model, path):
    """Trunk moments from the bundle, expert moments from each expert's file."""
    _load_expert_moments(opt, model, path)
    f = os.path.join(path, "optim.npz")
    if not os.path.exists(f):
        return
    z = np.load(f)
    name_of = {id(p): n for n, p in model.named_parameters()}
    for group in opt.param_groups:
        for p in group["params"]:
            n = name_of.get(id(p))
            if n is None or (n + "|m") not in z.files:
                continue
            m = unpack_bf16(z[n + "|m"])
            v = unpack_bf16(z[n + "|v"])
            # a tensor that changed shape since the moments were written - the
            # embedding after a vocabulary extension, an expert after growth -
            # has to start with fresh moments rather than mismatched ones
            if m.shape != p.shape or v.shape != p.shape:
                continue
            st = opt.state[p]
            st["exp_avg"] = m.to(device=p.device, dtype=p.dtype)
            st["exp_avg_sq"] = v.to(device=p.device, dtype=p.dtype)
            if (n + "|t") in z.files:
                # fused AdamW requires the step counter on the same device as
                # the parameter, not on the CPU where it was just loaded
                st["step"] = torch.tensor(float(z[n + "|t"]), device=p.device)


def _load_expert_moments(opt, model, path):
    name_of = {id(p): n for n, p in model.named_parameters()}
    cache = {}
    for group in opt.param_groups:
        for p in group["params"]:
            n = name_of.get(id(p))
            if n is None or not _is_expert(n):
                continue
            i, leaf = expert_index(n), expert_leaf(n)
            if i not in cache:
                f = os.path.join(path, EXPERTS, "e%05d.npz" % i)
                cache[i] = np.load(f) if os.path.exists(f) else None
            z = cache[i]
            if z is None or (leaf + "_m") not in z.files:
                continue
            m = unpack_bf16(z[leaf + "_m"])
            v = unpack_bf16(z[leaf + "_v"])
            if m.shape != p.shape:
                continue
            st = opt.state[p]
            st["exp_avg"] = m.to(device=p.device, dtype=p.dtype)
            st["exp_avg_sq"] = v.to(device=p.device, dtype=p.dtype)


def summarise(path):
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    ne = len([f for f in os.listdir(os.path.join(path, EXPERTS))
              if f.endswith((".npy", ".npz"))])
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    n_files = sum(len(fs) for _, _, fs in os.walk(path))
    print(f"{path}/  —  step {man.get('step')}, val {man.get('val')}")
    print(f"  experts/   {ne:>5} files   one per expert - the unit that is "
          f"paged, grown and pruned")
    print(f"  core.npz         1 file    {len(man['core_tensors'])} tensors: "
          f"embeddings, attention, norms")
    print(f"  routers.npz      1 file    {len(man['router_tensors'])} tensors: "
          f"one per call site")
    print(f"  optim.npz        1 file    Adam moments")
    print(f"  manifest.json    1 file    the index")
    print(f"  {n_files} files total, {tot/1e6:.1f} MB on disk")
    gates = [e["gate"] for e in man["experts"]]
    live = sum(1 for g in gates if abs(g) > 0.01)
    print(f"  {live} experts carrying weight, {len(gates)-live} gated to rest")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="weights")
    sys.exit(summarise(ap.parse_args().path))


# ----------------------------------------------------------------------------
# keeping the live model, and being able to get back
# ----------------------------------------------------------------------------

def best_val(path):
    """
    Validation loss of the state on disk, or inf if there is none.

    The directory holds the best state the model has reached, not the most
    recent one: `save` is called only when a run improves on it. A second copy
    kept aside as a snapshot would double a figure that is already 236 MB and
    grows with the pool, and it would be a copy of exactly what is already
    here.

    That makes the directory the thing to fall back to. A run that diverges
    reloads it and carries on, which costs the steps since the last
    improvement and nothing more.
    """
    f = os.path.join(path, "manifest.json")
    if not os.path.exists(f):
        return float("inf")
    with open(f) as fh:
        v = json.load(fh).get("val")
    return float(v) if v is not None else float("inf")
