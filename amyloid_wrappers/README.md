# amyloid-wrappers

Installable Python package that normalises per-residue outputs from amyloidogenicity
predictors into one schema (`position`, `aa_name`, `{Tool}_score`, `{Tool}_bin`) and
merges them into wide CSV tables for metascores and downstream analysis.

## Install

```bash
conda activate env   # or your venv
cd amyloid_wrappers
pip install -e ".[test]"    # package + pandas/numpy + pytest into THIS env
```

Verify the install uses the same Python:

```bash
which python pytest amyloid-parse
python -c "import amyloid_wrappers, pandas; print('OK')"
python -m pytest          # always uses env Python — preferred
```

### CLI commands

```bash
amyloid-wrappers --help          # overview of all commands
python -m amyloid_wrappers -h    # same

amyloid-parse --help             # parse one predictor
amyloid-merge --help             # merge standard CSVs
amyloid-run --help               # run PATH / APPNN (Phase 1)

python -m amyloid_wrappers parse --help
python -m amyloid_wrappers merge --help
python -m amyloid_wrappers run --help
```

### Troubleshooting pytest

| Symptom | Cause | Fix |
|---------|--------|-----|
| `No module named 'amyloid_wrappers'` | `pytest` is not from your env | `pip install -e ".[test]"` then `python -m pytest` |
| `No module named 'pandas'` | Same — system pytest (e.g. `apt install python3-pytest`) | Do **not** use `sudo apt install pytest`; use `pip install -e ".[test]"` |
| `amyloid-parse: command not found` | Package not installed in active env | `pip install -e ".[test]"` and check `which amyloid-parse` |

If `which pytest` shows `/usr/bin/pytest`, uninstall the apt package or always run:

```bash
python -m pytest
```

## Phase 0 checklist

| Goal | Status | Location |
|------|--------|----------|
| `PredictorResult` + wide merge | done | `core/schema.py`, `core/merge.py` |
| Parsers from notebook (+ PATH) | done | `predictors/*.py` |
| `amyloid-merge` CLI | done | `cli/merge.py` |
| Raw-output cache | done | `core/cache.py` |
| Predictor weights in config | done | `config/predictors.toml` |
| Golden tests vs BHT reference | done | `tests/test_golden.py` |
| PATH / APPNN runners | done | `runners/`, `cli/run.py`, `legacy/` |
| Web-tool runners (WALTZ, …) | phase 2–3 | `legacy/api/` reference scripts |

---

## Configuration (`config/predictors.toml`)

All tunable weights and thresholds live in one file. Override path:

```bash
export AMYLOID_WRAPPERS_CONFIG=/path/to/predictors.toml
amyloid-parse waltz ... --config /path/to/predictors.toml
```

### `[metascore.weights]`

Per-predictor weights for the linear metascore (sum must be **1.0**). Keys match the
registry (`path`, `appnn`, `crossbeta`, …). Used by `core/metascore.py` (phase 5).

Default: equal weights `0.125` × 8. To fit weights against existing hackathon tables:

```bash
python scripts/calibrate_weights.py \
  --merged-csv merged/RPS2_merged.csv \
  --metascore-csv ../BHT_amyloid/metascores/RPS2_metascore_table.csv
```

Paste the printed snippet into `[metascore.weights]`.

### `[predictors.*]`

Parser thresholds (binarisation cutoffs). CLI `--threshold` overrides for a single run.

### `[runners.path]` / `[runners.appnn]`

External tool paths for `amyloid-run`. Set `path.script` to `path1.1py` from
[PATH](https://github.com/KubaWojciechowski/PATH) or `AMYLOID_PATH_SCRIPT`.
APPNN defaults to `legacy/appnn_converter.R` in this package.

### `[cache]`

| Key | Default | Meaning |
|-----|---------|---------|
| `root` | `cache` | Base directory for raw copies |
| `enabled` | `true` | Set `false` or use `amyloid-parse --no-cache` to skip |

Layout: `cache/{protein_id}/{predictor}/raw.{ext}`

---

## Pipeline

```
FASTA / raw predictor output
        ↓  amyloid-run (PATH, APPNN) or manual tool + amyloid-parse
standard CSV per predictor   (+ optional raw cache)
        ↓  amyloid-merge
wide CSV (position, aa_name, all predictors)
        ↓  metascore (phase 5)
metascores/*_metascore.csv
```

---

## Canonical output format

All tools write the same per-residue schema:

| Column | Description |
|--------|-------------|
| `position` | 1-based residue index |
| `aa_name` | One-letter amino acid |
| `{Tool}_score` | Continuous score (see predictor section) |
| `{Tool}_bin` | 0/1 amyloid call |

Merged tables add columns from each predictor while keeping `position` and `aa_name`.

---

## CLI

### `amyloid-parse`

```bash
amyloid-parse waltz --input APP.dat --fasta APP.fasta -o APP_waltz.csv
amyloid-parse path --results results.csv --fasta RPS2.fasta -o RPS2_PATH.csv
```

| Flag | Purpose |
|------|---------|
| `--fasta` / `--sequence` | Sequence source (required) |
| `--input` / `--results` | Raw file (`--results` = PATH alias) |
| `--config` | TOML config path |
| `--threshold` | Override binarisation threshold |
| `--no-cache` | Skip raw cache copy |
| `--cache-dir` | Override `[cache].root` |

### `amyloid-run` (Phase 1)

```bash
amyloid-run appnn --fasta RPS2.fasta -o RPS2_APPNN.csv
amyloid-run path --fasta RPS2.fasta -o RPS2_PATH.csv --work-dir ./path_work

# Parse existing raw output without running the tool:
amyloid-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
amyloid-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv
```

PATH threading is slow — use `--skip-run` in CI or when `results.csv` already exists.

### `amyloid-merge`

```bash
amyloid-merge parsed/*.csv -o merged.csv --fasta RPS2.fasta
```

| Flag | Purpose |
|------|---------|
| `--predictor` | Force type per input file (if filename ambiguous) |
| `--fasta` | Validate identical sequence across inputs |

---

## Predictors: raw output → canonical mapping

Each parser implements `parse(source, protein_id, sequence) → PredictorResult`.
Thresholds default from `config/predictors.toml`.

### `path` — PATH threading

| | |
|---|---|
| **Raw input** | PATH `results.csv` (`seq`, `dope`, …) |
| **Algorithm** | Best (min) DOPE per hexapeptide → normalise to [0,1] with inversion → sliding window (6 aa) mean → per-residue score |
| **`PATH_score`** | Normalised per-residue score (0 = no hexapeptide coverage) |
| **`PATH_bin`** | 1 if score ≥ global percentile (default 75th of all hexapeptide scores) |
| **CLI** | `amyloid-run path …` or `amyloid-parse path --results results.csv …` |

### `appnn` — APPNN (R package output)

| | |
|---|---|
| **Raw input** | CSV from `BHT_amyloid/appnn_converter.R` |
| **Expected columns** | `aminoacid_position`, `aminoacid_score`, optional `aminoacid`, `hotspot_region` |
| **`APPNN_score`** | Per-residue APPNN score |
| **`APPNN_bin`** | 1 if `hotspot_region==1` or score ≥ 0.5 |
| **CLI** | `amyloid-parse appnn --input APP_APPNN.csv --fasta protein.fasta -o out.csv` |

### `waltz` — WALTZ standalone

| | |
|---|---|
| **Raw input** | Tab-separated `.dat` (no header): `position<TAB>score` |
| **`waltz_score`** | Tool score (0 if absent) |
| **`waltz_bin`** | 1 if score ≠ 0 |
| **CLI** | `amyloid-parse waltz --input protein.dat --fasta protein.fasta -o out.csv` |

### `pasta` — PASTA 2.0 energy profile

| | |
|---|---|
| **Raw input** | One numeric energy value per line (no header) |
| **`pasta_score`** | Raw PASTA energy (negative = more amyloid-prone) |
| **`pasta_bin`** | 1 if energy < −5 |
| **Note** | Legacy `*_all.csv` may contain re-normalised positive values; standard CSV keeps raw energy |

### `aggreprot` — AggreProt export

| | |
|---|---|
| **Raw input** | CSV with header row + columns `position`, `aggregation`, … |
| **`aggreprot_score`** | `aggregation` column |
| **`aggreprot_bin`** | 1 if aggregation ≥ 0.25 |
| **CLI** | `amyloid-parse aggreprot --input aggreprot.csv --fasta protein.fasta -o out.csv` |

### `archcandy` — ArchCandy regions

| | |
|---|---|
| **Raw input** | CSV columns `Start`, `Stop`, `Score` (case-insensitive) |
| **Algorithm** | For each region, set score = max region score on residues; bin = 1 inside any region |
| **`ArchCandy_score`** | Max region score covering residue (0 outside) |
| **`ArchCandy_bin`** | 1 if residue in any predicted region |

### `crossbeta` — Cross-Beta predictor (CRBM JSON)

| | |
|---|---|
| **Raw input** | JSON from CRBM datastore: `{id: [{AA_list: [{index, amino_acid, mean_confidence}]}]}` |
| **`cross-beta-predictor_score`** | `mean_confidence` per residue (`index` is 0-based in JSON → +1 for position) |
| **`cross-beta-predictor_bin`** | 1 if mean_confidence ≥ 0.5 |

### `aggrescan` — not implemented (phase 4)

Registered in schema for column names in merge only. Parser TBD.

---

## Python API

```python
from amyloid_wrappers.core.config import load_config
from amyloid_wrappers.core.cache import store_raw_cache
from amyloid_wrappers.core.merge import merge_predictor_tables, write_merge_csv
from amyloid_wrappers.core.metascore import compute_weighted_metascore
from amyloid_wrappers.predictors.registry import get_parser
from amyloid_wrappers.runners.registry import get_runner

cfg = load_config()
waltz = get_parser("waltz").parse("APP.dat", protein_id="APP", sequence=seq)
appnn = get_runner("appnn").run(fasta="RPS2.fasta", skip_run=True, raw_csv="…")

wide = merge_predictor_tables([waltz, appnn])
meta = compute_weighted_metascore(wide, config=cfg)
write_merge_csv([waltz, appnn], "merged.csv")
```

---

## Tests

```bash
python -m pytest
```

Do not rely on a system-wide `pytest` from `apt` — it uses a different Python than your conda/venv.

- Unit parsers: `tests/test_parsers.py`
- Cache: `tests/test_cache.py`
- Golden merge roundtrip vs `BHT_amyloid/all/RPS2_human_all.csv`
- Golden Cross-Beta vs `RPL27 and RPL36/Cross-beta predictor/RPL27.json`

---

## Package layout

```
amyloid_wrappers/
├── config/predictors.toml    weights, thresholds, cache, runners
├── legacy/                   frozen BHT reference scripts (see legacy/README.md)
├── scripts/calibrate_weights.py
├── src/amyloid_wrappers/
│   ├── core/                 schema, config, cache, fasta, merge, metascore
│   ├── predictors/           parse-only modules
│   ├── runners/              PATH / APPNN execution (Phase 1)
│   └── cli/                  parse, merge, run, app
└── tests/
```

---

## Relation to `BHT_amyloid/`

| Old file | Replacement |
|----------|-------------|
| `arch_cross_...ipynb` | `amyloid-parse` / `legacy/` copy |
| `path_converter.py` | `legacy/path_converter.py`, `runners/path.py`, `predictors/path.py` |
| `appnn_converter.R` | `legacy/appnn_converter.R`, `runners/appnn.py` |
| `all/*_all.csv` | wide merge via `amyloid-merge` (same score columns) |
