"""
Bring a Hugging Face dataset in as text this model can read.

Two shapes are handled. A dataset of conversations becomes turns wrapped in
the user and bot markers; a dataset of plain documents becomes the documents
themselves, separated by the end-of-text marker. Both are single symbols in
this alphabet rather than spellings.

Records are GROUPED into files rather than written one apiece. A million
conversations of a thousand characters would be a million files: ten times
their own size on disk, because each takes a filesystem block, and minutes
added to every startup. Grouping never splits a record, which is the property
that actually matters - a window must not span a seam that carries no meaning.

    python3 -m corpora fetch --dataset teknium/OpenHermes-2.5 \
        --kind chat --out data/train/chat/hermes
    python3 -m corpora fetch --dataset roneneldan/TinyStories \
        --kind text --out data/train/stories

Driven by `python3 -m corpora all`, which knows the datasets each lane wants.

Unless --keep-cache is set, this process gives `datasets` a temporary cache and
deletes only that directory afterwards. Converting OpenThoughts can leave
thirteen gigabytes of Arrow behind, but an unrelated Hugging Face cache is not
this script's to remove.
"""

import argparse
import os
import shutil
import sys
import tempfile

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def as_chat(row):
    turns = row.get("conversations") or row.get("messages") or []
    out = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        who = (t.get("from") or t.get("role") or "").lower()
        val = (t.get("value") or t.get("content") or "").strip()
        if not val:
            continue
        if who in ("system",):
            continue
        tag = "user" if who in ("user", "human") else "bot"
        out.append(f"<{tag}>\n{val}\n</{tag}>\n")
    text = "".join(out)
    return text if text.startswith("<user>") and "<bot>" in text else None


def as_text(row):
    t = (row.get("text") or row.get("content") or "").strip()
    return (t + "\n<|endoftext|>\n") if len(t) > 40 else None


KINDS = {"chat": as_chat, "text": as_text}


def _convert(a, load_dataset, cache_dir=None):
    kwargs = {"split": "train", "streaming": a.streaming}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    d = load_dataset(a.dataset, a.config, **kwargs)
    # A streamed dataset has no length - it is an iterator over a remote file,
    # and asking costs a full pass.
    print(f"  {len(d):,} records" if not a.streaming
          else "  record count unknown until the read finishes", flush=True)
    os.makedirs(a.out, exist_ok=True)

    conv = KINDS[a.kind]
    buf, files, chars, held, skipped = [], 0, 0, 0, 0
    def flush(where, idx):
        nonlocal chars
        if not buf:
            return
        text = "".join(buf)
        sub = where if where == a.held_out else os.path.join(
            where, f"{idx // a.shard:04d}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"part-{idx:06d}.txt"), "w") as f:
            f.write(text)
        chars += len(text)
        buf.clear()

    for i, row in enumerate(d):
        if a.limit and i >= a.limit:
            break
        t = conv(row)
        if t is None:
            skipped += 1
            continue
        buf.append(t)
        if sum(len(x) for x in buf) >= a.group:
            if a.held_out and held < a.hold:
                flush(a.held_out, held); held += 1
            else:
                flush(a.out, files); files += 1
            if (files + held) % 2000 == 0:
                print(f"    {files + held:,} files, {chars/1e6:,.0f}M "
                      f"characters", flush=True)
    if buf:
        flush(a.out, files); files += 1
    print(f"  wrote {files:,} files to {a.out}"
          + (f" and {held:,} to {a.held_out}" if a.held_out else "")
          + f", {chars/1e6:,.1f}M characters ({skipped:,} skipped)", flush=True)


def _remove_cache(path):
    """Remove the cache we created, or fail rather than claiming success."""
    shutil.rmtree(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None,
                    help="dataset configuration, e.g. 20231101.en for wikipedia")
    ap.add_argument("--streaming", action="store_true",
                    help="read the dataset over the network instead of "
                         "downloading it first. Wikipedia is tens of "
                         "gigabytes of Parquet and only the text is wanted, "
                         "so streaming it with --limit costs a fraction of "
                         "the disk and none of the wait.")
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", default=None,
                    help="where to keep some back; skipped if not given")
    ap.add_argument("--hold", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--group", type=int, default=16_000,
                    help="characters per file; a record is never split")
    ap.add_argument("--shard", type=int, default=2000,
                    help="files per subdirectory")
    ap.add_argument("--keep-cache", action="store_true",
                    help="use and preserve the normal shared datasets cache")
    a = ap.parse_args()

    from datasets import load_dataset
    what = a.dataset + (f" [{a.config}]" if a.config else "")
    print(f"  {'streaming' if a.streaming else 'downloading'} {what}",
          flush=True)
    cache_dir = None
    if not a.keep_cache:
        # Keep large Arrow intermediates on the same storage the corpus uses;
        # /tmp is often a small tmpfs and OpenThoughts alone can exceed it. The
        # hidden directory is ignored by the corpus walker if one surveys the
        # output while conversion is still running.
        os.makedirs(a.out, exist_ok=True)
        cache_root = os.path.realpath(a.out)
        cache_dir = tempfile.mkdtemp(prefix=".minagi-hf-datasets-",
                                     dir=cache_root)
    try:
        _convert(a, load_dataset, cache_dir)
    except BaseException as conversion_error:
        if cache_dir is not None:
            try:
                _remove_cache(cache_dir)
            except OSError as cleanup_error:
                raise RuntimeError(
                    f"dataset conversion failed ({conversion_error}); "
                    f"temporary cache cleanup also failed ({cleanup_error})"
                ) from conversion_error
        raise
    if cache_dir is not None:
        _remove_cache(cache_dir)
        print("  removed the temporary dataset cache", flush=True)

    if a.streaming:
        # datasets' streaming reader leaves a worker thread alive, and CPython
        # aborts in PyGILState_Release while finalising - AFTER every file is
        # written and the summary printed. The work is done and the exit code
        # is the only casualty, so leave before the interpreter tears down.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
