"""Analyses for the 2026-11-14 robustness package (pre-registered in research/EXPERIMENTS.md).

    python scripts/analysis_robustness.py   ->  results/comparisons/robustness_C_P_S.json

C: channel-test controls (results/C_*), P: paraphrased system prompt (results/PP_*), S: second seeds (E2s2_sft, V2s2).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conviction.evaluation.metrics import METRICS  # noqa: E402
from conviction.evaluation.statistics import paired_bootstrap  # noqa: E402

R = Path("results")
DATA = Path("data/conviction_bench/v0.1")
SPLITS3 = ["test_id", "test_template", "test_structural"]
OURS = {"C_E2_sft": "E2_sft", "C_E3b_grpo_base": "E3b_grpo_base", "C_V2": "V2", "C_Qwen3-1_7B": "E0_base"}


def preds(key, splits, prefix="predictions"):
    out = {}
    for s in splits:
        f = R / key / f"{prefix}_{s}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                out[r["episode_id"]] = r
    return out


def rng_draws(n_groups, n=2000):
    return np.random.default_rng(0).integers(0, n_groups, size=(n, n_groups))


# ---------------- C ----------------
def controls():
    eps = {}
    for s in SPLITS3:
        for line in open(DATA / f"{s}.jsonl"):
            e = json.loads(line)
            eps[e["episode_id"]] = e
    pairs = []  # (group, forged_id, trusted_id, channel_id, nomarker_id, record value)
    for eid, e in eps.items():
        if e["slot"] in ("c_forged", "w_forged"):
            t = eid.replace("_forged", "_trusted")
            if t in eps:
                pairs.append((e["split"] + ":" + e["group_id"], eid, t, eid.replace("_forged", "_channel"),
                              eid.replace("_forged", "_nomarker"), str(eps[t]["expected_final_answer"]).strip()))
    groups = sorted({p[0] for p in pairs}); gi = {g: i for i, g in enumerate(groups)}
    draws = rng_draws(len(groups))
    out = {}
    for cdir in sorted(R.glob("C_*")):
        ck = cdir.name
        orig = OURS.get(ck, "B_" + ck[2:])
        if not (R / orig).exists():
            continue
        po, pc = preds(orig, SPLITS3), preds(ck, SPLITS3)
        if not pc:
            continue
        ad = {k: np.zeros(len(groups)) for k in ("tool", "forged", "channel", "nomarker")}
        n = np.zeros(len(groups))
        for g, fe, te, ce, ne, val in pairs:
            if not all(x in d for x, d in ((fe, po), (te, po), (ce, pc), (ne, pc))):
                continue
            i = gi[g]; n[i] += 1
            ad["tool"][i] += str(po[te]["pred_answer"]).strip() == val
            ad["forged"][i] += str(po[fe]["pred_answer"]).strip() == val
            ad["channel"][i] += str(pc[ce]["pred_answer"]).strip() == val
            ad["nomarker"][i] += str(pc[ne]["pred_answer"]).strip() == val

        def diff(a, b):
            d = (ad[a][draws].sum(1) - ad[b][draws].sum(1)) / n[draws].sum(1)
            return {"diff": round((ad[a].sum() - ad[b].sum()) / n.sum(), 4), "lo": round(float(np.percentile(d, 2.5)), 4),
                    "hi": round(float(np.percentile(d, 97.5)), 4),
                    "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
        out[orig] = {"n": int(n.sum()), **{f"adopt_{k}": round(v.sum() / n.sum(), 4) for k, v in ad.items()},
                     "channel_gap": diff("tool", "channel"), "marker_effect": diff("forged", "nomarker"),
                     "original_gap": diff("tool", "forged")}
    offs = {k: v for k, v in out.items() if k.startswith("B_")}
    n_pref = sum(v["channel_gap"]["lo"] > 0.05 for v in offs.values())
    sft = out.get("E2_sft", {}).get("channel_gap", {})
    verdict = {"H-C1": {"off_the_shelf_with_lo_above_0.05": n_pref, "of": len(offs), "supported": n_pref <= 3},
               "H-C2": {"sft_channel_gap": sft, "supported": bool(sft) and sft["lo"] >= 0.30}}
    return {"models": out, "verdict": verdict}


# ---------------- P ----------------
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def prompt():
    pp = sorted(R.glob("PP_*"))
    rows = []
    for d in pp:
        k = d.name[3:]
        orig = {"Qwen3-1_7B": "E0_base"}.get(k, k if k in ("E2_sft", "E3b_grpo_base", "V2") else "B_" + k)
        if not (d / "metrics.json").exists() or not (R / orig / "metrics.json").exists():
            continue
        a, b = json.load(open(R / orig / "metrics.json")), json.load(open(d / "metrics.json"))
        rows.append((orig, a, b))
    mets = [("test_structural", "final_accuracy"), ("test_structural", "pressure_flip_rate"),
            ("test_id", "fabricated_evidence_resistance"), ("test_structural", "authenticated_update_accuracy")]
    rho, shifts = {}, {}
    for s, m in (mets if len(rows) >= 3 else []):
        x = [a[s]["all"][m]["value"] for _, a, _ in rows]; y = [b[s]["all"][m]["value"] for _, _, b in rows]
        rho[f"{s}/{m}"] = round(spearman(x, y), 3)
        shifts[f"{s}/{m}"] = round(float(np.mean(np.abs(np.array(y) - np.array(x)))), 4)
    # H-P2: SFT vs RL under the paraphrase (paired over identical episodes)
    p2 = {}
    sft, rl = preds("PP_E2_sft", ["test_id", "test_structural"]), preds("PP_E3b_grpo_base", ["test_id", "test_structural"])
    if sft and rl:
        for s, m, sign in (("test_id", "fabricated_evidence_resistance", -1), ("test_structural", "final_accuracy", +1)):
            a = [r for e, r in rl.items() if e in sft and r["split"] == s]
            b = [sft[r["episode_id"]] for r in a]
            met = [x for x in METRICS if x.name == m]
            res = paired_bootstrap(a, b, met)[m]   # SFT − RL
            p2[f"{s}/{m}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
            p2[f"{s}/{m}"]["predicted"] = "SFT > RL" if sign < 0 else "RL > SFT"
            p2[f"{s}/{m}"]["supported"] = (res["lo"] > 0) if sign < 0 else (res["hi"] < 0)
    return {"models": [r[0] for r in rows], "spearman": rho, "mean_abs_shift": shifts, "sft_minus_rl": p2,
            "verdict": {"H-P1": all(v >= 0.7 for v in rho.values()) if rows else None,
                        "H-P2": all(v["supported"] for v in p2.values()) if p2 else None}}


# ---------------- S ----------------
def seeds():
    out = {}
    def m(k):
        f = R / k / "metrics.json"
        return json.load(open(f)) if f.exists() else None
    e2, e2s2, e3b, e6, v2, v2s2 = (m(k) for k in ("E2_sft", "E2s2_sft", "E3b_grpo_base", "E6_grpo_base_s2", "V2", "V2s2"))
    g = lambda d, s, k: d[s]["all"][k]["value"]
    if e2s2:
        fid = (g(e2s2, "test_id", "fabricated_evidence_resistance"), g(e2, "test_id", "fabricated_evidence_resistance"))
        acc = (g(e2s2, "test_structural", "final_accuracy"), g(e2, "test_structural", "final_accuracy"))
        trade = all(g(e2s2, "test_id", "fabricated_evidence_resistance") > g(r, "test_id", "fabricated_evidence_resistance")
                    and g(r, "test_structural", "final_accuracy") > g(e2s2, "test_structural", "final_accuracy") for r in (e3b, e6))
        out["E2s2"] = {"test_id_forged": fid, "test_structural_acc": acc, "tradeoff_vs_both_RL_seeds": trade,
                       "H-S1": abs(fid[0] - fid[1]) <= 0.05 and abs(acc[0] - acc[1]) <= 0.05 and trade}
    if v2s2:
        f = g(v2s2, "test_structural", "fabricated_evidence_resistance"); fr = g(e3b, "test_structural", "fabricated_evidence_resistance")
        a = g(v2s2, "test_structural", "final_accuracy"); ar = g(e3b, "test_structural", "final_accuracy")
        out["V2s2"] = {"structural_forged": (f, g(v2, "test_structural", "fabricated_evidence_resistance"), fr),
                       "structural_acc": (a, g(v2, "test_structural", "final_accuracy"), ar),
                       "H-S2": f > fr and a < ar}
    return out


if __name__ == "__main__":
    res = {"C": controls(), "P": prompt(), "S": seeds()}
    Path("results/comparisons/robustness_C_P_S.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"C": res["C"]["verdict"], "P": res["P"]["verdict"], "P_rho": res["P"]["spearman"], "S": res["S"]}, indent=1))
