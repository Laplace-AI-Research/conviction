"""Figures, tables and number macros for the ConvictionBench paper, read from results/.

    uv run --extra analysis python paper/scripts/make_assets.py

Writes paper/figures/*.pdf, paper/tables/*.tex and paper/tables/numbers.tex. Every value comes from a file under
results/ (metrics.json, breakdown_*.json, train_info.json, comparisons/*.json, budget_ledger.json); nothing is typed
by hand. Run from the repository root.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT_FIG = ROOT / "paper" / "figures"
OUT_TAB = ROOT / "paper" / "tables"
sys.path.insert(0, str(ROOT / "scripts"))
from export_site import ADAPTER_NAMES, MODELS, NOTES  # noqa: E402  (single source for the leaderboard roster)

# Reference palette (dataviz skill, light mode): first three categorical slots for our adapters, grays for the rest.
C_SFT, C_RL, C_V2 = "#2a78d6", "#eb6834", "#1baf7a"
C_BASE, C_OTHER = "#3d3c39", "#a8a7a1"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ADAPTER_COLORS = {"E2_sft": C_SFT, "E3b_grpo_base": C_RL, "V2": C_V2}
OURS = {key: (ADAPTER_NAMES[key].split()[1], color) for key, color in ADAPTER_COLORS.items()}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.titlesize": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "lines.linewidth": 1.6,
    })


def jl(p):
    return json.loads(Path(p).read_text())


def metrics(key):
    return jl(R / key / "metrics.json")


def v(key, split, metric, view="all"):
    try:
        x = metrics(key)[split][view][metric]
    except KeyError:
        return None
    return x if x.get("value") is not None else None


def val(key, split, metric):
    x = v(key, split, metric)
    return None if x is None else x["value"]


def params_b(s):
    m = re.match(r"([\d.]+)B", s)
    return float(m.group(1)) if m else None


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def f2(x):
    return "--" if x is None else f"{x:.2f}"


def ci(x, d=3):
    if x is None:
        return "--"
    return f"{x['value']:.{d}f} {{\\scriptsize [{x['lo']:.2f}, {x['hi']:.2f}]}}"


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den


# ------------------------------------------------------------------------------------------------ number macros
NUM: dict[str, str] = {}


def num(name, x, d=3):
    NUM[name] = x if isinstance(x, str) else f"{x:.{d}f}"


def diffci(d, digits=3):
    s = lambda x: f"{x:+.{digits}f}".replace("-", "$-$")  # noqa: E731
    return f"{s(d['diff'])} [{s(d['lo'])}, {s(d['hi'])}]"


# ------------------------------------------------------------------------------------------------ tables
TRAIN_ROWS = [
    ("E0_base", "Base (Qwen3-1.7B)"), ("E1_guided", "+ guidance prompt"), ("E2_sft", "+ SFT"),
    ("E3_grpo", "+ SFT $\\to$ GRPO"), ("E3c_sft_control", "+ SFT $\\to$ more SFT (control)"),
    ("E3b_grpo_base", "+ GRPO only (seed 1)"), ("E6_grpo_base_s2", "+ GRPO only (seed 2)"),
    ("V2", "+ light SFT $\\to$ GRPO (v2)"),
]
TRAIN_COLS = [
    ("test_id", "final_accuracy"), ("test_id", "pressure_flip_rate"), ("test_id", "fabricated_evidence_resistance"),
    ("test_id", "insufficient_accuracy"), ("test_structural", "final_accuracy"),
    ("test_structural", "pressure_flip_rate"), ("test_structural", "legitimate_correction_rate"),
    ("test_structural", "state_changing_update_accuracy"), ("test_structural", "fabricated_evidence_resistance"),
]


def table_training():
    lines = [r"\begin{tabular}{l" + "c" * len(TRAIN_COLS) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{In-distribution (\texttt{test\_id})} & "
             r"\multicolumn{5}{c}{Held-out subtypes (\texttt{test\_structural})} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-10}",
             r"Condition & Final & Flip$\downarrow$ & Forged & Insuff. & Final & Flip$\downarrow$ & Corr. & "
             r"Update & Forged \\", r"\midrule"]
    best = {}
    for c in TRAIN_COLS:
        xs = [val(k, *c) for k, _ in TRAIN_ROWS if k not in ("E0_base", "E1_guided")]
        xs = [x for x in xs if x is not None]
        best[c] = min(xs) if c[1] == "pressure_flip_rate" else max(xs)
    for k, label in TRAIN_ROWS:
        cells = []
        for c in TRAIN_COLS:
            x = val(k, *c)
            s = f3(x)
            if x is not None and abs(x - best[c]) < 1e-9 and k not in ("E0_base", "E1_guided"):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if k in ("E1_guided", "E3c_sft_control", "E6_grpo_base_s2"):
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_TAB / "training.tex").write_text("\n".join(lines) + "\n")


def table_training_ci():
    """Appendix: every split, every primary metric, with CIs."""
    cols = [("final_accuracy", "Final"), ("clean_accuracy", "Clean"), ("pressure_flip_rate", "Flip$\\downarrow$"),
            ("legitimate_correction_rate", "Corr."), ("state_changing_update_accuracy", "Update"),
            ("fabricated_evidence_resistance", "Forged"), ("insufficient_accuracy", "Insuff."),
            ("false_abstention_rate", "F.abst.$\\downarrow$")]
    rows = TRAIN_ROWS + [("E2s2_sft", "+ SFT (seed 2)"), ("V2s2", "+ v2 (seed 2)")]
    out = []
    for split, name in [("test_id", "In-distribution (\\texttt{test\\_id})"),
                        ("test_template", "Held-out templates and framing (\\texttt{test\\_template})"),
                        ("test_structural", "Held-out subtypes (\\texttt{test\\_structural})"),
                        ("test_structural_natural", "Held-out subtypes, natural history (unpaired)")]:
        out += [r"\multicolumn{9}{l}{\textit{" + name + r"}} \\", r"\midrule"]
        for k, label in rows:
            if not (R / k / "metrics.json").exists() or split not in metrics(k):
                continue
            out.append(label + " & " + " & ".join(ci(v(k, split, m), 2) for m, _ in cols) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    head = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
            "Condition & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    (OUT_TAB / "training_full.tex").write_text("\n".join(head + out + [r"\end{tabular}"]) + "\n")


def leaderboard_rows():
    rows = []
    for key, label, lab, params, group in MODELS:
        if not (R / key / "metrics.json").exists():
            continue
        info = jl(R / key / "run_info.json")
        rows.append(dict(
            key=key, label=label.replace(" (ours)", ""), lab=lab, params=params, group=group,
            pb=params_b(params), engine=info.get("engine", "hf"), hf=info.get("model"),
            acc=val(key, "test_structural", "final_accuracy"), flip=val(key, "test_structural", "pressure_flip_rate"),
            corr=val(key, "test_structural", "legitimate_correction_rate"),
            upd=val(key, "test_structural", "state_changing_update_accuracy"),
            forged_id=val(key, "test_id", "fabricated_evidence_resistance"),
            forged_st=val(key, "test_structural", "fabricated_evidence_resistance"),
            insuff=val(key, "test_structural", "insufficient_accuracy"),
            fmt=val(key, "test_structural", "strict_format_rate"),
            nat_corr=v(key, "test_structural_natural", "legitimate_correction_rate"),
        ))
    return rows


def table_leaderboard(rows):
    head = [r"\begin{tabular}{llrcccccccc}", r"\toprule",
            r" & & & \multicolumn{6}{c}{Held-out subtypes (\texttt{test\_structural})} & In-dist. & \\",
            r"\cmidrule(lr){4-9}\cmidrule(lr){10-10}",
            r"Model & Lab & Params & Final & Flip$\downarrow$ & Corr. & Update & Forged & Insuff. & Forged & Fmt. \\",
            r"\midrule"]
    ours = [r for r in rows if r["group"].startswith("ours")]
    rest = sorted([r for r in rows if not r["group"].startswith("ours")], key=lambda r: -r["acc"])
    body = []
    for r in ours + [None] + rest:
        if r is None:
            body.append(r"\midrule")
            continue
        mark = ""
        if r["key"] in NOTES and not r["group"].startswith("ours"):
            mark = r"$^\dagger$"
        eng = r"$^\ast$" if r["engine"] == "vllm" else ""
        name = r["label"] + mark + eng
        if r["group"].startswith("ours"):
            name = r"\textbf{" + name + "}"
        lab = r["lab"].replace("·", "--").replace("&", r"\&")
        if r["group"].startswith("ours"):
            lab = "ours"
        body.append(" & ".join([name, lab, r["params"].replace(" MoE", "\\,MoE"), f3(r["acc"]), f3(r["flip"]),
                                f3(r["corr"]), f3(r["upd"]), f3(r["forged_st"]), f3(r["insuff"]),
                                f3(r["forged_id"]), f2(r["fmt"])]) + r" \\")
    (OUT_TAB / "leaderboard.tex").write_text("\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def table_models(rows):
    """Appendix: Hugging Face ids and engines."""
    body = [r"\begin{tabular}{lll}", r"\toprule", r"Model & Hugging Face id & Engine \\", r"\midrule"]
    for r in rows:
        if r["group"].startswith("ours"):
            continue
        hf = r["hf"].replace("_", r"\_")
        body.append(f"{r['label']} & \\texttt{{\\footnotesize {hf}}} & {'vLLM' if r['engine'] == 'vllm' else 'HF'} \\\\")
    (OUT_TAB / "model_ids.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def table_channel(rows):
    C = jl(R / "comparisons" / "robustness_C_P_S.json")["C"]["models"]
    lab = {r["key"]: r["label"] for r in rows}
    order = ["E2_sft", "V2", "E3b_grpo_base", "E0_base"]
    rest = sorted([k for k in C if k not in order and k in lab],
                  key=lambda k: -C[k]["channel_gap"]["diff"])
    body = [r"\begin{tabular}{lccccc}", r"\toprule",
            r" & \multicolumn{4}{c}{P(adopt record)} & \\", r"\cmidrule(lr){2-5}",
            r"Model & Tool & Forged & Pasted plain & No marker & Channel gap (tool $-$ plain) \\", r"\midrule"]
    for k in order + [None] + rest:
        if k is None:
            body.append(r"\midrule")
            continue
        c = C[k]
        name = lab.get(k, k)
        if k in OURS:
            name = r"\textbf{" + name + "}"
        body.append(" & ".join([name, f2(c["adopt_tool"]), f2(c["adopt_forged"]), f2(c["adopt_channel"]),
                                f2(c["adopt_nomarker"]), diffci(c["channel_gap"], 2)]) + r" \\")
    (OUT_TAB / "channel.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")


HOLM_LABELS = {  # internal run names -> the names the paper uses
    "H5 (E3 − E2)": r"SFT$\to$GRPO $-$ SFT", "H5′ (E3b − E2)": r"RL $-$ SFT (seed 1)",
    "H5′ replication (E6 − E2)": r"RL $-$ SFT (seed 2)", "H-T": r"Thinking on $-$ off",
    "H-V2a (V2 − E3b)": r"v2 $-$ RL", "H-C2 (SFT pure-channel gap)": r"SFT channel gap",
    "H-P2 (SFT − RL, paraphrase)": r"SFT $-$ RL (paraphrase)",
}


def table_holm():
    H = jl(R / "comparisons" / "holm.json")
    body = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"Hypothesis & Test & $\Delta$ [95\% CI] & $p$ & $p_\text{Holm}$ & Holm \\", r"\midrule"]
    for row in H["rows"]:
        test = row["test"].replace("_", r"\_")
        hyp = HOLM_LABELS.get(row["hypothesis"].split(" (")[0] if row["hypothesis"].startswith("H-T") else row["hypothesis"],
                             row["hypothesis"])
        d = {"diff": row["diff"], "lo": row["lo"], "hi": row["hi"]}
        mark = r"\checkmark" if row["significant_holm"] else ("(raw only)" if row["significant_raw"] else "")
        pv = "$<$0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
        body.append(f"{hyp} & \\texttt{{\\scriptsize {test}}} & {diffci(d)} & {pv} & "
                    f"{row['p_holm']:.3f} & {mark} \\\\")
    (OUT_TAB / "holm.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    num("holmN", str(H["n_tests"]))
    num("holmRaw", str(sum(r["significant_raw"] for r in H["rows"])))
    num("holmSig", str(sum(r["significant_holm"] for r in H["rows"])))


def table_thinking():
    T = jl(R / "comparisons" / "T_thinking_vs_nonthinking.json")["models"]
    ms = [("final_accuracy", "Final"), ("pressure_flip_rate", "Flip$\\downarrow$"),
          ("fabricated_evidence_resistance", "Forged"), ("authenticated_update_accuracy", "Update")]
    body = [r"\begin{tabular}{l" + "c" * len(ms) + "cc}", r"\toprule",
            "Model & " + " & ".join(f"{n} (off$\\to$on)" for _, n in ms) + r" & Think tok. & Unfin. \\", r"\midrule"]
    for name, t in T.items():
        cells = []
        for m, _ in ms:
            d = t["thinking_minus_nonthinking"][m]
            sig = "$^\\ast$" if (d["lo"] > 0 or d["hi"] < 0) else ""
            cells.append(f"{t['off'][m]:.2f}$\\to${t['on'][m]:.2f}{sig}")
        body.append(name + " & " + " & ".join(cells) + f" & {round(t['mean_gen_tokens'])} & {t['unfinished']} \\\\")
    (OUT_TAB / "thinking.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")


API = [("gpt-6-astra", "GPT-6 Astra"), ("gpt-6.1-sol", "GPT-6.1 Sol"), ("gpt-6.1-terra", "GPT-6.1 Terra"),
       ("gpt-6-luna", "GPT-6 Luna"), ("gpt-4.1", "GPT-4.1 (no reasoning)")]


def table_api():
    ms = ["final_accuracy", "pressure_flip_rate", "fabricated_evidence_resistance", "insufficient_accuracy"]
    body = [r"\begin{tabular}{lcccc}", r"\toprule",
            r"Model & Final (pub / priv) & Flip$\downarrow$ (pub / priv) & Forged (pub / priv) & "
            r"Insuff. (pub / priv) \\", r"\midrule"]
    maxgap = 0.0
    for mid, name in API:
        cells = []
        for m in ms:
            a, b = val(f"A_{mid}", "test_structural", m), val(f"AP_{mid}", "test_structural", m)
            maxgap = max(maxgap, abs(a - b))
            cells.append(f"{a:.2f} / {b:.2f}")
        body.append(name + " & " + " & ".join(cells) + r" \\")
    (OUT_TAB / "api.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    num("apiMaxGap", maxgap, 2)
    usd = sum(jl(R / f"{p}_{mid}" / "run_info.json")["usage_new"]["usd"] for mid, _ in API for p in ("A", "AP"))
    num("apiUSD", usd, 2)


def table_splits():
    man = jl(ROOT / "data" / "conviction_bench" / "v0.1" / "manifest.json")["splits"]
    body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Split & Groups & Episodes & \texttt{keep} & \texttt{revise} & \texttt{insufficient} & \texttt{answer} \\",
            r"\midrule"]
    tot = 0
    for s in ["train", "val", "test_id", "test_template", "test_structural"]:
        rows = [json.loads(l) for l in open(ROOT / "data" / "conviction_bench" / "v0.1" / f"{s}.jsonl")]
        dec = {}
        for r in rows:
            dec[r["expected_decision"]] = dec.get(r["expected_decision"], 0) + 1
        g = len({r["group_id"] for r in rows})
        tot += len(rows)
        body.append(f"\\texttt{{{s.replace('_', chr(92) + '_')}}} & {g:,} & {len(rows):,} & {dec.get('keep', 0):,} & "
                    f"{dec.get('revise', 0):,} & {dec.get('insufficient', 0):,} & {dec.get('answer', 0):,} \\\\")
    adv = sum(1 for _ in open(ROOT / "data" / "conviction_bench" / "v0.1" / "test_adversarial.jsonl"))
    body.append(f"\\texttt{{test\\_adversarial}} & -- & {adv} & \\multicolumn{{4}}{{c}}{{hand-written}} \\\\")
    (OUT_TAB / "splits.tex").write_text("\n".join(body + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    num("nEpisodes", f"{tot:,}".replace(",", "{,}"))
    num("nAdv", str(adv))
    _ = man


# ------------------------------------------------------------------------------------------------ figures
def fig_leaderboard(rows):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75))
    ax = axes[0]
    for r in rows:
        if r["group"].startswith("ours"):
            continue
        col = C_BASE if r["group"] == "base" else C_OTHER
        ax.scatter(r["acc"], r["flip"], s=16 if r["group"] != "base" else 26, color=col, edgecolor="white",
                   linewidth=0.6, zorder=3)
    for k, (name, col) in OURS.items():
        r = next(x for x in rows if x["key"] == k)
        ax.scatter(r["acc"], r["flip"], s=40, color=col, edgecolor="white", linewidth=0.8, zorder=4, marker="D")
    base = next(x for x in rows if x["group"] == "base")
    ax.annotate("base", (base["acc"], base["flip"]), xytext=(-4, 6), textcoords="offset points", fontsize=7,
                color=MUTED, ha="right")
    for k, (name, col) in OURS.items():
        r = next(x for x in rows if x["key"] == k)
        dx, dy = {"E2_sft": (3, 6), "E3b_grpo_base": (5, -9), "V2": (-6, 6)}[k]
        ax.annotate(name, (r["acc"], r["flip"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7, color=TEXT, ha="left" if dx > 0 else "right")
    ax.set_xlabel("Final accuracy, held-out subtypes")
    ax.set_ylabel("Pressure-flip rate (lower is better)")
    ax.set_xlim(0, 0.75); ax.set_ylim(0.3, 0.92)
    ax.set_title("(a) Accuracy vs. caving to pressure", loc="left")

    ax = axes[1]
    for r in rows:
        if r["group"].startswith("ours") or r["pb"] is None:
            continue
        col = C_BASE if r["group"] == "base" else C_OTHER
        ax.scatter(r["pb"], r["forged_id"], s=16 if r["group"] != "base" else 26, color=col, edgecolor="white",
                   linewidth=0.6, zorder=3)
    for k, (name, col) in OURS.items():
        r = next(x for x in rows if x["key"] == k)
        ax.scatter(r["pb"], r["forged_id"], s=40, color=col, edgecolor="white", linewidth=0.8, zorder=4, marker="D")
        ax.annotate(name, (r["pb"], r["forged_id"]), xytext=(6, -3),
                    textcoords="offset points", fontsize=7, color=TEXT)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 14]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "14"])
    ax.minorticks_off()
    ax.set_xlabel("Parameters (B, log scale)")
    ax.set_ylabel("Forged-record resistance, in-dist.")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) Resisting records pasted into user turns", loc="left")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="D", ls="", color=c, label=n, markeredgecolor="white", markersize=6)
         for n, c in OURS.values()]
    h += [Line2D([], [], marker="o", ls="", color=C_BASE, label="Qwen3-1.7B (base)", markersize=5),
          Line2D([], [], marker="o", ls="", color=C_OTHER, label="Off-the-shelf (37)", markersize=5)]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_FIG / "leaderboard.pdf"); plt.close(fig)


def fig_tradeoff():
    """SFT vs RL vs v2 vs base on the metrics where they trade off."""
    ms = [("test_id", "fabricated_evidence_resistance", "Forged-record\nresist. (in-dist.)"),
          ("test_id", "insufficient_accuracy", "Abstention\n(in-dist.)"),
          ("test_structural", "final_accuracy", "Final acc.\n(held-out)"),
          ("test_structural", "state_changing_update_accuracy", "Tool-update acc.\n(held-out)"),
          ("test_structural_natural", "legitimate_correction_rate", "Accepts correction\nof own error (held-out)")]
    conds = [("E0_base", "Base", C_BASE), ("E2_sft", *OURS["E2_sft"]),
             ("E3b_grpo_base", *OURS["E3b_grpo_base"]), ("V2", *OURS["V2"])]
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    w = 0.19
    for i, (k, name, col) in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for j, (s, m, _) in enumerate(ms):
            x = v(k, s, m)
            xs.append(j + (i - 1.5) * (w + 0.015)); ys.append(x["value"])
            lo.append(x["value"] - x["lo"]); hi.append(x["hi"] - x["value"])
        ax.bar(xs, ys, width=w, color=col, label=name, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=0.6, capsize=1.5, zorder=4)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([m[2] for m in ms], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
    fig.savefig(OUT_FIG / "tradeoff.pdf"); plt.close(fig)


def fig_channel(rows):
    C = jl(R / "comparisons" / "robustness_C_P_S.json")["C"]["models"]
    lab = {r["key"]: r["label"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    show = [("E0_base", "Base", C_BASE), ("E2_sft", *OURS["E2_sft"]),
            ("V2", *OURS["V2"]), ("E3b_grpo_base", *OURS["E3b_grpo_base"])]
    variants = [("adopt_tool", "Tool channel"), ("adopt_channel", "Pasted plain"), ("adopt_forged", "Forged wording")]
    shades = ["#3d3c39", "#8a8984", "#d4d3ce"]
    w = 0.25
    for j, (fld, vname) in enumerate(variants):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(show))]
        ys = [C[k][fld] for k, _, _ in show]
        ax.bar(xs, ys, width=w, color=shades[j], label=vname, zorder=3)
    ax.set_xticks(range(len(show))); ax.set_xticklabels([s[1] for s in show])
    for t, (_, _, col) in zip(ax.get_xticklabels(), show):
        t.set_color(col if col != C_BASE else TEXT)
    ax.set_ylabel("P(adopts the record's value)"); ax.set_ylim(0, 0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)
    ax.set_title("(a) Same record, three deliveries", loc="left")

    ax = axes[1]
    keys = [k for k in C if k in lab]
    keys.sort(key=lambda k: C[k]["channel_gap"]["diff"])
    for i, k in enumerate(keys):
        g = C[k]["channel_gap"]
        col = OURS[k][1] if k in OURS else (C_BASE if k == "E0_base" else C_OTHER)
        ax.plot([g["lo"], g["hi"]], [i, i], color=col, lw=1.0, zorder=3)
        ax.scatter(g["diff"], i, s=14 if k not in OURS else 26, color=col, zorder=4, edgecolor="white", linewidth=0.5,
                   marker="D" if k in OURS else "o")
        if k in OURS or k == "E0_base":
            ax.annotate({"E0_base": "base", **{kk: vv[0] for kk, vv in OURS.items()}}[k],
                        (g["hi"], i), xytext=(3, -2.5), textcoords="offset points", fontsize=6.5, color=TEXT)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.axvline(0.05, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks([]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Channel gap: P(adopt | tool) $-$ P(adopt | pasted plain)")
    ax.set_title(f"(b) Channel gap, {len(keys)} models (95% CI)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "channel.pdf"); plt.close(fig)


def fig_thinking():
    T = jl(R / "comparisons" / "T_thinking_vs_nonthinking.json")["models"]
    ms = [("final_accuracy", "Final acc."), ("pressure_flip_rate", "Pressure flip"),
          ("authenticated_update_accuracy", "Tool update"), ("fabricated_evidence_resistance", "Forged resist.")]
    names = list(T)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    shades = ["#2a78d6", "#6aa2e6", "#eb6834", "#f2a17f"]
    h = 0.18
    for i, n in enumerate(names):
        for j, (m, _) in enumerate(ms):
            d = T[n]["thinking_minus_nonthinking"][m]
            y = j + (i - 1.5) * (h + 0.02)
            ax.barh(y, d["diff"], height=h, color=["#3d3c39", "#6e6d69", "#9e9d98", "#c9c8c2"][i], zorder=3,
                    label=n if j == 0 else None)
            ax.plot([d["lo"], d["hi"]], [y, y], color=TEXT, lw=0.6, zorder=4)
    _ = shades
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([m[1] for m in ms]); ax.invert_yaxis()
    ax.set_xlabel("Thinking on $-$ off (paired, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=2)
    fig.savefig(OUT_FIG / "thinking.pdf"); plt.close(fig)


def fig_dynamics():
    runs = [("E3_grpo", "SFT$\\to$GRPO", C_SFT), ("E3b_grpo_base", "GRPO only, seed 1", C_RL),
            ("E6_grpo_base_s2", "GRPO only, seed 2", "#f2a17f"), ("V2", "v2 (light SFT$\\to$GRPO)", C_V2)]
    series = [("reward", "Mean reward"), ("frac_reward_zero_std", "Zero-variance groups"),
              ("completions/mean_length", "Completion length (tokens)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1))
    for k, name, col in runs:
        p = R / k / "train_info.json"
        if not p.exists():
            continue
        hist = [h for h in jl(p)["log_history"] if "reward" in h]
        steps = [h["step"] for h in hist]
        for ax, (m, title) in zip(axes, series):
            ys = [float(h[m]) for h in hist]
            roll = [sum(ys[max(0, i - 9): i + 1]) / len(ys[max(0, i - 9): i + 1]) for i in range(len(ys))]
            ax.plot(steps, ys, color=col, lw=0.5, alpha=0.25)
            ax.plot(steps, roll, color=col, lw=1.6, label=name)
            ax.set_title(title, loc="left"); ax.set_xlabel("GRPO step")
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_FIG / "dynamics.pdf"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ prose numbers
def numbers(rows):
    off = [r for r in rows if r["group"] in ("other", "base")]
    shelf = [r for r in rows if r["group"] == "other"]
    num("nModels", str(len(rows)))
    num("nShelf", str(len(shelf)))
    num("nLabs", str(len({r["lab"] for r in shelf})))
    num("nShelfForgedLow", str(sum(r["forged_id"] <= 0.25 for r in shelf)))
    flips = sorted(r["flip"] for r in shelf)
    num("shelfFlipMin", flips[0], 2); num("shelfFlipMax", flips[-1], 2)
    best = max(shelf, key=lambda r: r["acc"])
    num("bestShelfName", best["label"]); num("bestShelfAcc", best["acc"])
    sp = [r for r in shelf if r["pb"]]
    lp = [math.log(r["pb"]) for r in sp]
    num("rhoForged", spearman(lp, [r["forged_st"] for r in sp]), 2)
    num("rhoAcc", spearman(lp, [r["acc"] for r in sp]), 2)
    num("rhoUpd", spearman(lp, [r["upd"] for r in sp]), 2)
    num("rhoAccUpd", spearman([r["acc"] for r in sp], [r["upd"] for r in sp]), 2)
    _ = off

    for key, short in [("E0_base", "Base"), ("E1_guided", "Guided"), ("E2_sft", "SFT"), ("E3_grpo", "SFTRL"),
                       ("E3c_sft_control", "Ctrl"), ("E3b_grpo_base", "RL"), ("E6_grpo_base_s2", "RLs"),
                       ("V2", "Vtwo"), ("E2s2_sft", "SFTs"), ("V2s2", "Vtwos")]:
        for split, sname in [("test_id", "Id"), ("test_template", "Tpl"), ("test_structural", "St"),
                             ("test_structural_natural", "Nat")]:
            for m, mname in [("final_accuracy", "Acc"), ("clean_accuracy", "Clean"), ("pressure_flip_rate", "Flip"),
                             ("social_pressure_flip_rate", "SocFlip"),
                             ("misleading_derivation_flip_rate", "DerivFlip"),
                             ("legitimate_correction_rate", "Corr"), ("state_changing_update_accuracy", "Upd"),
                             ("fabricated_evidence_resistance", "Forged"), ("insufficient_accuracy", "Insuff"),
                             ("capitulation_rate", "Cap"), ("wrong_prior_recovery_rate", "Recov"),
                             ("strict_format_rate", "Fmt")]:
                x = v(key, split, m)
                if x is not None:
                    num(f"{short}{sname}{mname}", x["value"])
                    if split == "test_structural_natural":
                        NUM[f"{short}{sname}{mname}N"] = str(x["n"])

    comp = R / "comparisons"
    e = jl(comp / "E3b_vs_E2.json")
    print("E3b_vs_E2 keys:", list(e.keys())[:6])

    # robustness
    rb = jl(comp / "robustness_C_P_S.json")
    C = rb["C"]["models"]
    shelfkeys = [k for k in C if k.startswith("B_") or k.startswith("C_")]
    shelfkeys = [k for k in C if k not in ("E0_base", "E2_sft", "E3b_grpo_base", "V2")]
    num("chanN", str(len(shelfkeys)))
    num("chanAbove", str(sum(C[k]["channel_gap"]["lo"] > 0.05 for k in shelfkeys)))
    num("chanNeg", str(sum(C[k]["channel_gap"]["diff"] < 0 for k in shelfkeys)))
    num("chanNegSig", str(sum(C[k]["channel_gap"]["hi"] < 0 for k in shelfkeys)))
    num("markerMax", max(abs(C[k]["marker_effect"]["diff"]) for k in C), 2)
    for k, s in [("E2_sft", "SFT"), ("V2", "Vtwo"), ("E3b_grpo_base", "RL"), ("E0_base", "Base")]:
        c = C[k]
        num(f"chan{s}Tool", c["adopt_tool"], 2); num(f"chan{s}Plain", c["adopt_channel"], 2)
        num(f"chan{s}Forged", c["adopt_forged"], 2); num(f"chan{s}NoMarker", c["adopt_nomarker"], 2)
        num(f"chan{s}Gap", diffci(c["channel_gap"], 2)); num(f"chan{s}OrigGap", diffci(c["original_gap"], 2))
    P = rb["P"]
    for kk, s in [("test_structural/final_accuracy", "Acc"), ("test_structural/pressure_flip_rate", "Flip"),
                  ("test_id/fabricated_evidence_resistance", "Forged"),
                  ("test_structural/authenticated_update_accuracy", "Upd")]:
        num(f"rho{s}Para", P["spearman"][kk], 2)
        num(f"shift{s}Para", P["mean_abs_shift"][kk], 2)
    num("paraForgedDiff", diffci(P["sft_minus_rl"]["test_id/fabricated_evidence_resistance"]))
    num("paraAccDiff", diffci(P["sft_minus_rl"]["test_structural/final_accuracy"]))
    num("nPara", str(len(P["models"])))

    for name, fn in [("RLvsSFT", "E3b_vs_E2.json"), ("RLsvsSFT", "E6_vs_E2.json"), ("SFTRLvsSFT", "E3_vs_E2.json"),
                     ("SFTRLvsCtrl", "E3_vs_control.json"), ("GuidedvsBase", "E1_vs_E0.json"),
                     ("RLsvsRL", "E6_vs_E3b.json")]:
        d = jl(comp / fn)
        sp = d.get("splits", d)
        for split, sname in [("test_id", "Id"), ("test_template", "Tpl"), ("test_structural", "St")]:
            block = sp.get(split)
            if not isinstance(block, dict):
                continue
            if len(block) == 1 and isinstance(next(iter(block.values())), dict) and \
                    "final_accuracy" in next(iter(block.values())):
                block = next(iter(block.values()))
            for m, mname in [("final_accuracy", "Acc"), ("pressure_flip_rate", "Flip"),
                             ("fabricated_evidence_resistance", "Forged"), ("insufficient_accuracy", "Insuff"),
                             ("state_changing_update_accuracy", "Upd"), ("legitimate_correction_rate", "Corr"),
                             ("social_pressure_flip_rate", "SocFlip"), ("capitulation_rate", "Cap")]:
                if m in block and isinstance(block[m], dict) and "diff" in block[m]:
                    num(f"d{name}{sname}{mname}", diffci(block[m]))
    V = jl(comp / "V2_vs_E3b_E2.json")["splits"]
    num("dVtwoRLIdForged", diffci(V["test_id"]["V2_minus_E3b"]["fabricated_evidence_resistance"]))
    num("dVtwoRLStAcc", diffci(V["test_structural"]["V2_minus_E3b"]["final_accuracy"]))
    VN = jl(comp / "V2_vs_E3b_E2.json")["natural"]
    num("dVtwoSFTNatCorr", diffci(VN["V2_minus_E2_natural_legit"]))

    # training dynamics summaries
    def zstd(k, first=False):
        h = [x for x in jl(R / k / "train_info.json")["log_history"] if "reward" in x]
        xs = [x["frac_reward_zero_std"] for x in h]
        return sum(xs) / len(xs)
    num("zstdSFTRL", zstd("E3_grpo") * 100, 0)
    num("zstdRL", zstd("E3b_grpo_base") * 100, 0)
    num("zstdVtwo", zstd("V2") * 100, 0)

    led = jl(R / "budget_ledger.json")
    num("gpuUSD", sum(e["cost"] for e in led["entries"]), 2)

    lines = ["% Generated by paper/scripts/make_assets.py from results/. Do not edit by hand."]
    for k, x in sorted(NUM.items()):
        lines.append(f"\\newcommand{{\\n{k}}}{{{x}}}")
    (OUT_TAB / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True); OUT_TAB.mkdir(parents=True, exist_ok=True)
    style()
    rows = leaderboard_rows()
    table_training(); table_training_ci(); table_leaderboard(rows); table_models(rows); table_channel(rows)
    table_holm(); table_thinking(); table_api(); table_splits()
    fig_leaderboard(rows); fig_tradeoff(); fig_channel(rows); fig_thinking(); fig_dynamics()
    numbers(rows)
    print(f"{len(rows)} models; wrote {len(list(OUT_FIG.glob('*.pdf')))} figures, "
          f"{len(list(OUT_TAB.glob('*.tex')))} tables, {len(NUM)} number macros")


if __name__ == "__main__":
    main()
