# ConvictionBench paper (Laplace Research edition)

```bash
uv run --extra analysis python paper/scripts/make_assets.py   # figures, tables, numbers.tex from results/
cd paper && tectonic -X compile main.tex                       # -> paper/main.pdf
```

Every number in the text is a macro from `tables/numbers.tex` or a table under `tables/`, both generated from
`results/` by `scripts/make_assets.py`. The exceptions, quoted from `research/EXPERIMENTS.md`: the policy-audit
rewards (0.89 / 0.66 / 0.53 / 0.20), the pre-freeze response-format A/B (0.19 → 0.56), the parity-gate maximum
(0.017), the E3b training-reward range (0.39 → 0.71), and the second RL seed's Holm p-value (0.072, also in
`results/comparisons/holm.json`).

## Before posting
- [x] Human validation stated as a limitation (planned for v0.2).
- [x] All bib entries verified online (2026-11-17).
- [x] GPT API-track results kept as Appendix G.
- [ ] Sync the release PDF to `convictionbench/paper.pdf` in the Laplace website repository, `paper.pdf` in the benchmark website repository, and `papers/convictionbench.pdf` in the personal website repository. Keep the existing `paper-laplace.pdf` and `convictionbench-laplace.pdf` URLs in sync.
- [ ] arXiv: build with `tectonic -X compile --keep-intermediates main.tex` to get `main.bbl`, then upload `main.tex`,
      `main.bbl`, `tables/`, `figures/`; pick categories (cs.CL primary, cs.LG).
