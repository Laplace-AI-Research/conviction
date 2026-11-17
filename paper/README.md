# ConvictionBench paper (arXiv preprint)

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
- [ ] Human validation: fill the Limitations TODO (search `\todo`) with agreement and label accuracy for the 160 items.
- [ ] Spot-check the bib entries not yet verified online (listed at the top of `refs.bib`).
- [ ] Decide whether to keep the GPT API-track appendix (currently Appendix G).
- [ ] arXiv: build with `tectonic -X compile --keep-intermediates main.tex` to get `main.bbl`, then upload `main.tex`,
      `main.bbl`, `tables/`, `figures/`; pick categories (cs.CL primary, cs.LG).
