"""Holm–Bonferroni over every pre-registered confirmatory test with a paired-bootstrap p-value (pre-registered
2026-11-14, research/EXPERIMENTS.md § M). p_boot = 0 is floored at 1/2000 (the bootstrap's resolution).
Rule-based hypotheses (counts, non-inferiority margins, unpaired CI comparisons) have no single p-value and are listed
separately with their verdicts.

    python scripts/holm.py  ->  results/comparisons/holm.json
"""
import json
from pathlib import Path

C = Path("results/comparisons")
PRIMARY = ["final_accuracy", "clean_accuracy", "pressure_flip_rate", "legitimate_correction_rate",
           "authenticated_update_accuracy", "fabricated_evidence_resistance", "insufficient_accuracy", "false_abstention_rate"]
FLOOR = 1 / 2000


def load(name):
    p = C / name
    return json.loads(p.read_text()) if p.exists() else None


def tests():
    t = []
    e3 = load("E3_vs_E2.json")
    for s in ("test_template", "test_structural"):
        for m in PRIMARY:
            x = e3["splits"][s][m]
            t.append(("H5 (E3 − E2)", f"{s}/{m}", x))
    for f, h in (("E3b_vs_E2.json", "H5′ (E3b − E2)"), ("E6_vs_E2.json", "H5′ replication (E6 − E2)")):
        d = load(f)
        for m in ("final_accuracy", "pressure_flip_rate"):
            t.append((h, f"test_structural/{m}", d["splits"]["test_structural"][m]))
    th = load("T_thinking_vs_nonthinking.json")
    for model, v in th["models"].items():
        for m in ("fabricated_evidence_resistance", "pressure_flip_rate"):
            t.append(("H-T (thinking − off)", f"{model}/{m}", v["thinking_minus_nonthinking"][m]))
    v2 = load("V2_vs_E3b_E2.json")
    t.append(("H-V2a (V2 − E3b)", "test_id/fabricated_evidence_resistance",
              v2["splits"]["test_id"]["V2_minus_E3b"]["fabricated_evidence_resistance"]))
    rob = load("robustness_C_P_S.json")
    if rob:
        sft = rob["C"]["models"].get("E2_sft", {}).get("channel_gap")
        if sft:
            t.append(("H-C2 (SFT pure-channel gap)", "tool − pasted plain", sft))
        for k, v in rob["P"].get("sft_minus_rl", {}).items():
            t.append(("H-P2 (SFT − RL, paraphrase)", k, v))
    return t


def main():
    rows = []
    for h, what, x in tests():
        if x.get("p_boot") is None:
            continue
        rows.append({"hypothesis": h, "test": what, "diff": round(x["diff"], 4), "lo": round(x["lo"], 4),
                     "hi": round(x["hi"], 4), "p": max(x["p_boot"], FLOOR)})
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i]["p"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i]["p"])
        running = max(running, adj)
        rows[i]["p_holm"] = round(running, 4)
        rows[i]["significant_holm"] = running < 0.05
        rows[i]["significant_raw"] = rows[i]["p"] < 0.05
    out = {"n_tests": m, "alpha": 0.05, "rows": rows,
           "lost_after_correction": [f"{r['hypothesis']}: {r['test']}" for r in rows if r["significant_raw"] and not r["significant_holm"]]}
    (C / "holm.json").write_text(json.dumps(out, indent=1))
    print(f"{m} tests; significant raw {sum(r['significant_raw'] for r in rows)}, after Holm {sum(r['significant_holm'] for r in rows)}")
    for s in out["lost_after_correction"]:
        print("  lost:", s)


if __name__ == "__main__":
    main()
