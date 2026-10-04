"""The four Ma'at gates as local, deterministic tools.

A reference port of the gate logic — not a call to an external service — so the
model can run Screen / Judge / Weigh / Council without depending on the MCP
transport. maat_screen and maat_judge reproduce the observed Mekhat MCP
behaviour; maat_weigh and council_weigh are simplified deterministic versions.
Wire the live MCP in later via a generic HTTP tool for the full council scaffold.
"""

from .tools import Tool


def maat_screen(text, context=None):
    t = (text or "").strip()
    if not t:
        return {"band": "Yellow", "reasons": ["empty"]}
    return {"band": "Green", "reasons": []}


def _gate(verdict, evidence, reason):
    return {"gate": None, "verdict": verdict, "evidence": evidence,
            "reason": reason}


def maat_judge(name, is_verified=False, is_open_source=False,
               has_lock_in=False, is_proportional=False,
               is_maintainable=False, security_scan_result="unknown",
               evidence_level="assumption"):
    def g(gate, verdict, evidence, reason):
        d = _gate(verdict, evidence, reason)
        d["gate"] = gate
        return d

    gates = []
    gates.append(g("TRUTH",
                   "PASS" if is_verified else "FAIL",
                   evidence_level,
                   "" if is_verified else
                   f"'{name}' is unverified. Run a test or confirm it works "
                   "before adopting."))
    gates.append(g("FAIR_SHARE",
                   "PASS" if is_open_source and not has_lock_in else "FAIL",
                   f"open_source={is_open_source}, lock_in={has_lock_in}",
                   "" if (is_open_source and not has_lock_in) else
                   f"'{name}' is not open-source. Document why the lock-in is "
                   "acceptable."))
    gates.append(g("RIGHT_SIZE",
                   "PASS" if is_proportional else "FAIL",
                   "",
                   "" if is_proportional else
                   f"'{name}' is not proportional to the problem. Reduce scope "
                   "or justify."))
    gates.append(g("LASTING_VALUE",
                   "PASS" if is_maintainable else "FAIL",
                   "",
                   "" if is_maintainable else
                   f"'{name}' has no clear maintenance path. Document "
                   "sustainment plan."))
    sec_verdict = ("PASS" if security_scan_result == "clean" else
                   "DEFER" if security_scan_result == "unknown" else "FAIL")
    gates.append(g("SECURITY",
                   sec_verdict,
                   security_scan_result,
                   "" if sec_verdict == "PASS" else
                   "No security scan. Run SkillSpector first."))
    # Simplified port: a per-gate DEFER (e.g. an unknown security scan) is
    # flattened to FAIL here; wire the live MCP for true three-valued defer.
    verdict = "PASS" if all(x["verdict"] == "PASS" for x in gates) else "FAIL"
    return {"verdict": verdict, "gates": gates}


def maat_weigh(claim, evidence_level="assumption", has_tests=False,
               is_documented=False, is_reversible=False):
    truth = evidence_level in ("test_success", "formally_verified")
    fair = True
    size = True
    lasting = is_documented and is_reversible
    ok = truth and fair and size and lasting
    return {"verdict": "PASS" if ok else "FAIL",
            "truth": "PASS" if truth else "FAIL",
            "fair_share": "PASS" if fair else "FAIL",
            "right_size": "PASS" if size else "FAIL",
            "lasting_value": "PASS" if lasting else "FAIL",
            "tests": has_tests}


def council_weigh(proposal):
    md = (f"# The Weighing\n\n**Proposal:** {proposal}\n\n"
          f"**Ma'at screen:** Green\n\n"
          f"*Simplified local port. Wire the live MCP via an HTTP tool for the "
          f"full lens/seat/ledger scaffold.*\n")
    return {"band": "Green", "blocked": False, "markdown": md}


def register_maat_tools(registry):
    registry.register(Tool("maat_screen", "screen text through the Ma'at Screen gate",
                           {"text": "str", "context": "str"},
                           lambda a: maat_screen(a.get("text"), a.get("context"))))
    registry.register(Tool("maat_judge", "judge whether a dependency/tool should be adopted",
                           {}, lambda a: maat_judge(
                               a.get("name"), a.get("is_verified", False),
                               a.get("is_open_source", False),
                               a.get("has_lock_in", False),
                               a.get("is_proportional", False),
                               a.get("is_maintainable", False),
                               a.get("security_scan_result", "unknown"),
                               a.get("evidence_level", "assumption"))))
    registry.register(Tool("maat_weigh", "deep four-test Ma'at gate on a claim",
                           {}, lambda a: maat_weigh(
                               a.get("claim"), a.get("evidence_level", "assumption"),
                               a.get("has_tests", False),
                               a.get("is_documented", False),
                               a.get("is_reversible", False))))
    registry.register(Tool("council_weigh", "full council weighing of a proposal",
                           {}, lambda a: council_weigh(a.get("proposal"))))
