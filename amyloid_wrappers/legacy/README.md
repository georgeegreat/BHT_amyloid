# Legacy reference scripts (from BHT_amyloid)

Frozen copies of hackathon scripts kept inside the package for traceability and
runner integration. **Do not import these modules at runtime** — use
`amyloid_wrappers.predictors` and `amyloid_wrappers.runners` instead.

| File | Original role | Package replacement |
|------|---------------|---------------------|
| `path_converter.py` | PATH batch + APR export | `predictors/path.py`, `runners/path.py` |
| `appnn_converter.R` | APPNN R runner | `runners/appnn.py` (invokes this script) |
| `arch_cross_pasta_aggreprot_waltz_parser.ipynb` | Notebook parsers | `predictors/*.py` |
| `parse_predictor.py` | Thin CLI shim | `amyloid-parse` |
| `api/aggreprot.py` | Selenium runner | Phase 2–3 |
| `api/cross_candy.py` | CRBM web runner | Phase 2–3 |
| `api/PASTA 2.0.py` | PASTA web runner | Phase 2–3 |

Source of truth for edits: update wrappers first; refresh legacy copies when the
reference script changes intentionally.
