# EXP12 Output Audit Before EXP13

Inspected files from `outputs/exp12_independent_replication`:

- summary.json
- replication_summary.csv
- replication_contrasts.csv
- family_summary.csv
- specificity_comparison.csv
- replication_results.csv
- token_anchor_audit.csv
- run_manifest.json

Verified observations that materially changed EXP13:

1. `frozen_V_sufficiency = -0.013089` and `frozen_V_necessity = -0.014992`.
2. Negative-offset controls are positive (+0.010903 / +0.012403).
3. Target readers remain positive (+0.076307 / +0.061981), whereas negative
   readers are strongly negative (-0.084568 / -0.105234).
4. Target-minus-negative reader contrasts are +0.160875 / +0.167215.
5. all7 reconstruction/removal is exact, non-KV0 leakage is zero.
6. Frozen offset tokens are identical across all family/wording conditions.
7. Canonical-vs-paraphrase frozen-V sufficiency differs systematically:
   about +0.053 config_command, +0.048 docs_code, +0.048 search_edit,
   +0.014 test_edit.
8. Several family/wording/label cells reverse sign, so writer portability is
   not bidirectional under the frozen write sites.
9. The reader register remains positive in every family and both wording
   variants, although its magnitude is weaker for paraphrases.

These observations motivate family×wording writer discovery and an explicit
bidirectionality constraint in EXP13.
