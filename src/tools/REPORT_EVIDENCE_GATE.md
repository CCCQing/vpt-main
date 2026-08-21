# Experiment report evidence gate

`src/tools/report_evidence_gate.py` makes report completeness a machine-checked state instead of a manually declared conclusion.

Each experiment owns two input files:

- `report_evidence_profile.json`: artifact sources, discovery scope, report path, generated files, and automatic tables;
- `report_coverage_manifest.json`: rules mapping every atomic evidence group to a report section or a justified non-reporting state.

The tool generates three audit artifacts next to the report:

- `atomic_evidence_inventory.json`: atomic groups derived from real artifacts;
- `report_coverage_resolved.json`: the rule selected for every atomic group;
- `report_acceptance.json`: the final receipt, hashes, errors, and `report_may_claim_complete` flag.

Run:

```bash
python src/tools/report_evidence_gate.py sync --profile <report_evidence_profile.json>
python src/tools/report_evidence_gate.py check --profile <report_evidence_profile.json>
python src/tools/report_evidence_gate.py check-tree --root <experiment-analysis-root>
```

`sync` rebuilds the inventory, generated Markdown tables, status block, and receipt. `check` is read-only and fails when artifacts, coverage, report headings, generated tables, or the receipt have drifted.

`sync-tree` and `check-tree` scan every report named `实验结果分析.md` below a root. A report without its own `report_evidence_profile.json` fails the tree gate, so adding a new report cannot silently bypass completeness validation.

The atomic unit for `method_named_metric_summary.csv(.gz)` is:

```text
evidence_role × checkpoint × split × condition × domain × entity_type × probe_selection_seed
```

Individual Prompt, head, attribute, class, sample, and layer identities remain traceable through entity counts, examples, and hashes without forcing every low-level row into the report.

Coverage rules must match each atomic group exactly once. A rule declares one of:

```text
presented | audit_only | detail_omitted | not_applicable |
not_requested | legacy_not_available | invalid
```

Unmapped groups, ambiguous rules, missing report sections, unregistered discovered artifacts, unavailable required sources, edited generated tables, stale inventories, or stale receipts fail the gate. If the gate fails, `report_may_claim_complete=false`; a report completeness claim is itself an additional validation error.

Automatic tables are delimited by `report-evidence:<marker>:start/end`. They must never be edited by hand. `coverage_summary` and `source_summary` are available for audit tables. `metric_summary` can produce mean and min-to-max tables directly from the method-level summary by exact selectors.
