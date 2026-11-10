#!/usr/bin/env python3
"""
The sample log's readings, and the history drawn from it.

What this asserts:

  --sample-readings, or samples.readings in config.yaml, takes `written` and
  `raw`, as a list or separated by commas, and refuses anything else or
  nothing

  a `written` reading is written with config.yaml's decoding as the run
  started, a `raw` one plain greedy, and the log names the written one's rule

  the log's header gives greedy's repeated 8-grams on the line their history
  has always been read from, and the written reading's on a line of its own
  that names the rule; tools/export_history.py puts both on the record they
  follow

Every repeated sequence here is built in code rather than written out, so
this file shows the self-knowledge lane no loop to learn from.
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train as T                                              # noqa: E402
import export_history as EH                                    # noqa: E402

ok = True


def check(name, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra else ""))


print("readings")
check("written and raw, separated by commas",
      T._readings("written, raw") == ["written", "raw"])
check("or as a list", T._readings(["raw"]) == ["raw"])
for bad in ("adapted", "", []):
    try:
        T._readings(bad)
        refused = False
    except argparse.ArgumentTypeError:
        refused = True
    check(f"{bad!r} is refused", refused)

args = argparse.Namespace(decoding={"dry": 0.8, "dry_allowed": 16},
                          sample_readings=["written", "raw"])
check("written is decoding's settings, raw is greedy",
      T._sample_variants(args) == [("written", {"dry": 0.8, "dry_allowed": 16}),
                                   ("raw", {})])
rule = T._rule(args)
check("the written reading's rule, in a line",
      rule == "greedy, DRY 0.8 x 1.15^(repeat - 16)", repr(rule))

print("the log and its history")
loop = "ab" * 40                         # 71 of its 73 8-grams seen before
plain = "the quick brown fox jumps over the lazy dog"
tmp = tempfile.mkdtemp(prefix="minagi-readings-")
saved = dict(T._PLOTS)
T._PLOTS["on"] = False                   # no plotters for a test's log
try:
    log = os.path.join(tmp, "samples.txt")
    for k in (1, 2):
        T.write_samples(log, 10 * k, k * 1e6, k,
                        [("stories", "Once", [("written", plain), ("raw", loop)])],
                        val=0.6, se=0.01, rule=rule)
    text = open(log, encoding="utf-8").read()
    check("greedy's line reads as it always has",
          "repeats 97% of 8-grams, greedy with no guard" in text)
    check("the written reading's line names its rule",
          f"written {rule}: repeats 0% of 8-grams" in text)
    rows = EH.parse(log)
    check("both land on the record they follow",
          len(rows) == 2 and all(r.get("repeats_pct") == 97.0
                                 and r.get("written_repeats_pct") == 0.0
                                 for r in rows),
          str([(r.get("repeats_pct"), r.get("written_repeats_pct")) for r in rows]))
finally:
    T._PLOTS.clear()
    T._PLOTS.update(saved)
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
