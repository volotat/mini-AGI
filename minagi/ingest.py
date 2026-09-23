"""
Reading a pile of files.

The model's alphabet is the 256 byte values, so there is nothing to prepare: a
file is already written in the only vocabulary it has. Point it at a directory
and it reads what is there - source, prose, notes, logs, transcripts - in
whatever order they come, and is different afterwards.

That is the whole interface. No tokenizer to fit, no corpus to build, no
preprocessing step that has to be re-run when the data changes.

Binary files are skipped rather than read as noise: a file is taken as text if
a sample of it decodes as UTF-8 and is mostly printable. Everything else is
left alone, because a model that reads a JPEG learns what a JPEG header looks
like and nothing else useful.
"""

import codecs
import hashlib
import json
import os
import re

import numpy as np

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
             "venv", ".mypy_cache", ".pytest_cache", "build", "dist"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
            ".gz", ".bz2", ".xz", ".tar", ".7z", ".mp3", ".mp4", ".wav",
            ".mov", ".avi", ".so", ".dylib", ".dll", ".exe", ".bin", ".npy",
            ".npz", ".pt", ".pth", ".onnx", ".pyc", ".woff", ".woff2", ".ttf"}


def looks_like_text(path, probe=4096):
    """Is this worth reading? Decides on a sample, not on the extension alone."""
    if os.path.splitext(path)[1].lower() in SKIP_EXT:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(probe)
    except OSError:
        return False
    if not head:
        return False
    if b"\x00" in head:
        return False
    # Decode the sample as a STREAM, so a multi-byte character cut in half by
    # the end of the probe is held rather than raised on. The probe ends on an
    # arbitrary byte, so it lands inside a UTF-8 sequence whenever the text
    # happens to put one there - and calling the file binary for that drops
    # ordinary text from the corpus.
    try:
        s = codecs.getincrementaldecoder("utf-8")().decode(head)
    except UnicodeDecodeError:
        return False
    if not s:
        return False
    printable = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    return printable / len(s) > 0.9


def _walk(paths, follow_symlinks=False):
    """Every candidate path, without opening anything."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
            continue
        for root, dirs, files in os.walk(p, followlinks=follow_symlinks):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not d.startswith(".")]
            for f in files:
                out.append(os.path.join(root, f))
    return sorted(out)


_CACHE_KEEP = 6          # distinct corpora remembered at once


def collect(paths, follow_symlinks=False, cache="runs/corpus_index.json"):
    """
    Every readable text file under the given files and directories.

    Deciding whether a file is text means reading a piece of it, and at half a
    million files that is three minutes of every startup. Walking the names is
    fast; opening them is not. So the answer is remembered, keyed on the names
    themselves - if the same set of paths comes back, the same verdicts hold,
    and nothing needs to be opened.

    A file whose contents changed under an unchanged name keeps its old
    verdict, which is the price. Text does not usually turn into something
    else, and the corpus tools write new names rather than overwrite.
    """
    names = _walk(paths, follow_symlinks)
    sig = hashlib.md5(
        ("\n".join(names) + "|" + "|".join(sorted(paths))).encode()
    ).hexdigest()
    # Several corpora are surveyed in one session - the training set, the
    # held-out set, a scratch directory under test - so the cache keeps one
    # entry per signature rather than one entry in total. A single slot meant
    # each survey evicted the last, and a run that alternated between two
    # directories paid the full three minutes every single time.
    held = {}
    if cache and os.path.exists(cache):
        try:
            with open(cache) as f:
                got = json.load(f)
            held = got.get("entries") or {}
            if sig in held:
                return held[sig]
        except (OSError, ValueError, KeyError, AttributeError):
            held = {}
    out = sorted({p for p in names if looks_like_text(p)})
    if cache:
        try:
            held[sig] = out
            # keep the largest few: a corpus of half a million files is what
            # the cache exists for, and a scratch directory of three is not
            # worth evicting it over
            if len(held) > _CACHE_KEEP:
                for k in sorted(held, key=lambda k: len(held[k]),
                                reverse=True)[_CACHE_KEEP:]:
                    del held[k]
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            tmp = cache + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"entries": held}, f)
            os.replace(tmp, cache)
        except OSError:
            pass
    return out


_TOK = None

# `surrogateescape` gives a byte that is not valid UTF-8 a lone surrogate of
# its own, and re-encoding a surrogate with the tokenizer's strict codec
# raises. A file is a byte sequence, and reading one may not fail on the bytes
# it happens to contain, so those runs are spliced back as the bytes they came
# from while everything around them goes through the tokenizer - which is what
# keeps the structural markers single ids.
_LONE = re.compile("[\udc80-\udcff]+")


def _ids(text):
    """`text` as token ids: total over every byte sequence it could hold."""
    out, pos = [], 0
    for m in _LONE.finditer(text):
        out.extend(_TOK.encode(text[pos:m.start()]).ids)
        out.extend(text[m.start():m.end()].encode("utf-8", "surrogateescape"))
        pos = m.end()
    out.extend(_TOK.encode(text[pos:]).ids)
    return out


def as_stream(path):
    """
    One file as the character stream the model reads.

    Encoded through the tokenizer rather than read as raw bytes. For ordinary
    text the two are the same - every byte is its own token. They differ only
    where a file contains one of the structural markers, `<user>` or `<g>` or
    `<think>`: those are single ids above 255, and reading the file as bytes
    would turn each into six or seven separate characters and lose the
    boundary it marks.
    """
    global _TOK
    if _TOK is None:
        from minagi.tokenizer import ByteTokenizer
        _TOK = ByteTokenizer()
    with open(path, "rb") as f:
        raw = f.read()
    return np.asarray(_ids(raw.decode("utf-8", "surrogateescape")),
                      dtype=np.uint16)


def summarise(files):
    total = sum(os.path.getsize(f) for f in files)
    return {"files": len(files), "characters": total}
