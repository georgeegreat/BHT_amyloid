# amyloid-wrappers (aka Wrappers for AGGRESSOR)

Installable Python package that normalises per-residue outputs from amyloidogenicity  
predictors into one schema (`position`, `aa_name`, `{Tool}_score`, `{Tool}_bin`) and  
merges them into wide CSV tables for metascores and downstream analysis.

## Install

Recommended: one conda env **`wrappers`** with the package, PATH tooling, and test deps.
Python-only deps are declared in `pyproject.toml`; Modeller comes from conda; PyRosetta
and APPNN (R) are installed once per machine (see below).

### New environment

```bash
cd amyloid_wrappers
conda env create -f environment.yml   # Python 3.11 + Modeller; pip deps from requirements.txt
conda activate wrappers
pip install -e ".[test]"
```

### Existing `wrappers` env (install / upgrade missing pieces)

```bash
conda activate wrappers
conda env update -f environment.yml --prune
pip install -e ".[test]"
```

`pip install -e .` pulls core deps from `pyproject.toml` (`pandas`, `numpy`, `biopython`,
`scikit-learn`). Re-running it is safe: only missing or outdated wheels are installed.

### Verify

```bash
which python pytest amyloid-parse amyloid-run
python -c "import amyloid_wrappers, Bio, sklearn; print('OK')"
python -m pytest
```

### External tools (once per machine)

| Tool | Install | Used by |
|------|---------|---------|
| **Modeller** | `conda install -c salilab modeller` (in `environment.yml`) | PATH (`vendor/PATH/path1.1.py`) |
| **PyRosetta** | `pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release` or [pyrosetta.org](https://www.pyrosetta.org/) | PATH |
| **APPNN** (R) | R + CRAN package `appnn` (+ `dplyr`, `tidyr`, `readr`, `stringr`, `purrr`); runner calls `legacy/appnn_converter.R` | APPNN |

**APPNN (R) install** — if system site-library is not writable, packages go to the user library automatically:

```bash
Rscript -e 'install.packages(c("appnn","dplyr","tidyr","readr","stringr","purrr"), repos="https://cloud.r-project.org")'
Rscript -e 'library(appnn); cat("appnn OK\n")'
```

**Modeller license** (required for PATH live runs):

```bash
export KEY_MODELLER="YOUR_LICENSE_KEY"    # add to ~/.bashrc
```

Edit `$CONDA_PREFIX/lib/modeller-10.8/modlib/modeller/config.py`:

```python
install_dir = r'/path/to/anaconda3/envs/wrappers/lib/modeller-10.8'
license = r'YOUR_LICENSE_KEY'
```

Tested pip/conda versions (env `wrappers`, Python 3.11): numpy 2.4.6, pandas 3.0.3,
biopython 1.87, scikit-learn 1.9.0, modeller 10.8, pyrosetta 2026.3+releasequarterly.

### CLI commands

```bash
amyloid-wrappers --help          # overview of all commands
python -m amyloid_wrappers -h    # same

amyloid-parse --help             # parse one predictor
amyloid-merge --help             # merge standard CSVs
amyloid-run --help               # run PATH / APPNN (Phase 1)
amyloids-widemerge --help        # merge + optional BHT reference check

python -m amyloid_wrappers batch --help   # multifasta batch pipeline
python wrappers_run.py --help             # same (root script)

python -m amyloid_wrappers parse --help
python -m amyloid_wrappers merge --help
python -m amyloid_wrappers run --help
python -m amyloid_wrappers widemerge --help
```

## Phase 0 checklist


| Goal                           | Status    | Location                            |
| ------------------------------ | --------- | ----------------------------------- |
| `PredictorResult` + wide merge | done      | `core/schema.py`, `core/merge.py`   |
| Parsers from notebook (+ PATH) | done      | `predictors/*.py`                   |
| `amyloid-merge` CLI            | done      | `cli/merge.py`                      |
| Raw-output cache               | done      | `core/cache.py`                     |
| Predictor weights in config    | done      | `config.cfg`            |
| Golden tests vs BHT reference  | done      | `tests/test_golden.py`              |
| PATH / APPNN runners           | done      | `runners/`, `vendor/PATH/`, `legacy/` |
| Metascore weight presets       | done      | `config.cfg` |
| `amyloids-widemerge`           | done      | `cli/widemerge.py` |
| Web-tool runners (WALTZ, …)    | phase 2–3 | `legacy/api/` reference scripts     |


---

## Configuration (`config.cfg`)

All tunable weights and thresholds live in one INI-style `.cfg` file. Override path:

```bash
export AMYLOID_WRAPPERS_CONFIG=/path/to/config.cfg
amyloid-parse waltz ... --config /path/to/config.cfg
```

### Metascore weight presets

Three presets in `config.cfg` (7 predictors; AggreProt excluded for now):

| Preset | Use case |
|--------|----------|
| `functional_amyloids` | WALTZ/PATH/APPNN/PASTA emphasis |
| `pathogenic_amyloids` | Cross-Beta/APPNN emphasis |
| `predictor_specificity` | **default** — conservative, tool-specificity oriented |

Active preset: `[metascore].preset` or env `AMYLOID_METASCORE_PRESET`.

Optional least-squares fit against hackathon tables:

```bash
python scripts/calibrate_weights.py \
  --merged-csv merged/RPS2_merged.csv \
  --metascore-csv ../BHT_amyloid/metascores/RPS2_metascore_table.csv
```

Paste the printed snippet into a new preset table under `[metascore.presets.*]`.

### `[predictors.*]`

Parser thresholds (binarisation cutoffs). CLI `--threshold` overrides for a single run.

### `[runners.path]` / `[runners.appnn]`

PATH uses bundled `vendor/PATH/path1.1.py` (default script). Run inside the **`wrappers`**
conda env so the same Python has Modeller + PyRosetta.

```ini
[runners.path]
script =
python = python3
```

Overrides: `AMYLOID_PATH_SCRIPT`, `AMYLOID_PATH_PYTHON`.

APPNN defaults to `legacy/appnn_converter.R` (`Rscript` on `PATH`).

#### Vendored PATH (vs [upstream](https://github.com/KubaWojciechowski/PATH))

Wojciechowski & Kotulska, *Sci Rep* **10**, 7721 (2020). Local tweaks in `path1.1.py`:
auto-detect Modeller binary, `--modeller` flag, sklearn pickle compat, skip finished
Modeller runs, clearer empty-results error.

---

### `[cache]`


| Key       | Default | Meaning                                               |
| --------- | ------- | ----------------------------------------------------- |
| `root`    | `cache` | Base directory for raw copies                         |
| `enabled` | `false` | Set `true` or pass CLI `--keep-cache` to retain cache |


Layout: `cache/{protein_id}/{predictor}/raw.{ext}` — removed after each run unless `--keep-cache`.

---



## Pipeline

```
FASTA / raw predictor output
        ↓  amyloid-run (PATH, APPNN) or manual tool + amyloid-parse
standard CSV per predictor   (+ optional raw cache)
        ↓  amyloid-merge  (or amyloids-widemerge with --reference)
wide CSV (position, aa_name, all predictors)
        ↓  metascore (phase 5)
metascores/*_metascore.csv
```

---



## Canonical output format

All tools write the same per-residue schema:


| Column         | Description                              |
| -------------- | ---------------------------------------- |
| `position`     | 1-based residue index                    |
| `aa_name`      | One-letter amino acid                    |
| `{Tool}_score` | Continuous score (see predictor section) |
| `{Tool}_bin`   | 0/1 amyloid call                         |


Merged tables add columns from each predictor while keeping `position` and `aa_name`.

---

## CLI

### `amyloid-parse`

```bash
amyloid-parse waltz --input APP.dat --fasta APP.fasta -o APP_waltz.csv
amyloid-parse path --results results.csv --fasta RPS2.fasta -o RPS2_PATH.csv
```


| Flag                     | Purpose                             |
| ------------------------ | ----------------------------------- |
| `--fasta` / `--sequence` | Sequence source (required)          |
| `--input` / `--results`  | Raw file (`--results` = PATH alias) |
| `--config`               | CFG config path (`config.cfg`) |
| `--threshold`            | Override binarisation threshold     |
| `--keep-cache`           | Keep `cache/` after run (default: remove) |
| `--cache-dir`            | Override `[cache].root`             |


### `amyloid-run` (Phase 1)

```bash
amyloid-run appnn --fasta RPS2.fasta -o RPS2_APPNN.csv
amyloid-run path --fasta RPS2.fasta -o RPS2_PATH.csv --work-dir ./path_work

# Parse existing raw output without running the tool:
amyloid-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
amyloid-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv
```

PATH threading is slow — use `--skip-run` in CI or when `results.csv` already exists.

### `amyloid-wrappers batch` (multifasta PATH + APPNN)

Runs every sequence in a multifasta through external runners, parses to standard CSV,
and writes one wide merged table per protein. Progress: `[PATH]`, `[APPNN]`, `[merge]`.

```bash
conda activate wrappers
python -m amyloid_wrappers batch vendor/PATH/test.fasta -o output_dir \
    --predictors path,appnn --batch-size 3
# or: python wrappers_run.py vendor/PATH/test.fasta -o output_dir
```

**Output layout** (`-o` / `--output`):

```
output_dir/
├── PATH/parsed/{protein_id}_PATH.csv
├── APPNN/parsed/{protein_id}_APPNN.csv
├── merged/{protein_id}_merged.csv
└── .tmp/                  # removed after successful run (fasta split scratch)
```

Per-predictor `work/` directories (Modeller/PyRosetta/R temp files) are **removed
automatically** after a successful run. Use `--save-raw-files DIR` to archive raw
`{protein_id}/{predictor}.*` copies (needed for `--skip-run` after work cleanup).

| Flag | Purpose |
| ---- | ------- |
| `-o` / `--output` | Output root (creates `PATH/`, `APPNN/`, `merged/` subdirs) |
| `--predictors` | Comma-separated list (default: `path,appnn`) |
| `--batch-size` | Sequences per APPNN invocation (PATH always uses 1) |
| `--save-raw-files` | Archive raw tool outputs per protein (enables `--skip-run` after cleanup) |
| `--keep-cache` | Keep `cache/` after run (default: removed) |
| `--skip-run` | Parse only; read raw from `{PREDICTOR}/work/` or `--save-raw-files` |

**Smoke test** (3 short proteins, ~4–5 min with PATH live run):

```bash
python -m amyloid_wrappers batch vendor/PATH/test.fasta -o output_dir --predictors path,appnn
ls output_dir/PATH/parsed output_dir/APPNN/parsed output_dir/merged
```

### `amyloid-merge`

```bash
amyloid-merge parsed/*.csv -o merged.csv --fasta RPS2.fasta
```


| Flag          | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `--predictor` | Force type per input file (if filename ambiguous) |
| `--fasta`     | Validate identical sequence across inputs         |

### `amyloids-widemerge`

Same merge as `amyloid-merge`, plus optional regression check against a BHT
`*_all.csv` reference table (score columns only).

```bash
amyloids-widemerge --input-dir parsed/ --fasta RPS2.fasta -o merged/RPS2_merged.csv
amyloids-widemerge parsed/*.csv --fasta RPS2.fasta -o merged.csv \
    --reference ../all/RPS2_human_all.csv
amyloids-widemerge --input-dir parsed/ -o merged.csv --check-bht-reference \
    --reference-name RPS2_human_all.csv
```

| Flag | Purpose |
| ---- | ------- |
| `--input-dir` | Directory of standard CSVs (or pass paths as arguments) |
| `--reference` | BHT wide table for score-column comparison |
| `--check-bht-reference` | Default reference under `BHT_amyloid/all/` |
| `--rtol` / `--atol` | Tolerance for reference comparison |

Also available as `python -m amyloid_wrappers widemerge …`.

---

## Predictors: raw output → canonical mapping

Each parser implements `parse(source, protein_id, sequence) → PredictorResult`.
Thresholds default from `config.cfg`.

### `path` — PATH threading


|               |                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Raw input** | PATH `results.csv` (`seq`, `dope`, …)                                                                                |
| **Algorithm** | Best (min) DOPE per hexapeptide → normalise to [0,1] with inversion → sliding window (6 aa) mean → per-residue score |
| `PATH_score`  | Normalised per-residue score (0 = no hexapeptide coverage)                                                           |
| `PATH_bin`    | 1 if score ≥ global percentile (default 75th of all hexapeptide scores)                                              |
| **CLI**       | `amyloid-run path …` or `amyloid-parse path --results results.csv …`                                                 |




### `appnn` — APPNN (R package output)


|                      |                                                                                 |
| -------------------- | ------------------------------------------------------------------------------- |
| **Raw input**        | CSV from `legacy/appnn_converter.R` (`APPNN_parsed/{id}_APPNN.csv`) |
| **Expected columns** | `aminoacid_position`, `aminoacid_score`, optional `aminoacid`, `hotspot_region` |
| `APPNN_score`        | Per-residue APPNN score                                                         |
| `APPNN_bin`          | 1 if `hotspot_region==1` or score ≥ 0.5                                         |
| **CLI**              | `amyloid-parse appnn --input APP_APPNN.csv --fasta protein.fasta -o out.csv`    |




### `waltz` — WALTZ standalone


|               |                                                                            |
| ------------- | -------------------------------------------------------------------------- |
| **Raw input** | Tab-separated `.dat` (no header): `position<TAB>score`                     |
| `waltz_score` | Tool score (0 if absent)                                                   |
| `waltz_bin`   | 1 if score ≠ 0                                                             |
| **CLI**       | `amyloid-parse waltz --input protein.dat --fasta protein.fasta -o out.csv` |




### `pasta` — PASTA 2.0 energy profile


|               |                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------- |
| **Raw input** | One numeric energy value per line (no header)                                               |
| `pasta_score` | Raw PASTA energy (negative = more amyloid-prone)                                            |
| `pasta_bin`   | 1 if energy < −5                                                                            |
| **Note**      | Legacy `*_all.csv` may contain re-normalised positive values; standard CSV keeps raw energy |




### `aggreprot` — AggreProt export


|                   |                                                                                  |
| ----------------- | -------------------------------------------------------------------------------- |
| **Raw input**     | CSV with header row + columns `position`, `aggregation`, …                       |
| `aggreprot_score` | `aggregation` column                                                             |
| `aggreprot_bin`   | 1 if aggregation ≥ 0.25                                                          |
| **CLI**           | `amyloid-parse aggreprot --input aggreprot.csv --fasta protein.fasta -o out.csv` |




### `archcandy` — ArchCandy regions


|                   |                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------ |
| **Raw input**     | CSV columns `Start`, `Stop`, `Score` (case-insensitive)                              |
| **Algorithm**     | For each region, set score = max region score on residues; bin = 1 inside any region |
| `ArchCandy_score` | Max region score covering residue (0 outside)                                        |
| `ArchCandy_bin`   | 1 if residue in any predicted region                                                 |




### `crossbeta` — Cross-Beta predictor (CRBM JSON)


|                              |                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Raw input**                | JSON from CRBM datastore: `{id: [{AA_list: [{index, amino_acid, mean_confidence}]}]}` |
| `cross-beta-predictor_score` | `mean_confidence` per residue (`index` is 0-based in JSON → +1 for position)          |
| `cross-beta-predictor_bin`   | 1 if mean_confidence ≥ 0.5                                                            |


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

## Package layout

```
amyloid_wrappers/
├── config.cfg                weights, thresholds, cache, runners
├── environment.yml           conda env `wrappers` (Python + Modeller)
├── requirements.txt          pip runtime dependencies
├── requirements-dev.txt      pytest, ruff, build
├── wrappers_run.py           batch pipeline entry point
├── vendor/PATH/              vendored PATH (path1.1.py, templates, models)
├── legacy/                   frozen BHT reference scripts
├── scripts/calibrate_weights.py
├── src/amyloid_wrappers/
│   ├── batch/                multifasta pipeline
│   ├── core/                 schema, config, cache, fasta, merge, validate, metascore
│   ├── predictors/           parse-only modules
│   ├── runners/              PATH / APPNN execution (Phase 1)
│   └── cli/                  parse, merge, run, batch, widemerge, app
└── tests/                    unit + golden tests (fixtures only)
```

---

## Tests

```bash
python -m pytest
```

!!Do not rely on a system-wide `pytest` from `apt` — it uses a different Python than your conda/venv!!

- Unit parsers: `tests/test_parsers.py`
- Cache: `tests/test_cache.py`
- Batch: `tests/test_batch.py`
- Golden merge roundtrip vs `BHT_amyloid/all/RPS2_human_all.csv`
- Golden Cross-Beta vs `RPL27 and RPL36/Cross-beta predictor/RPL27.json`

---

## Manual multifasta workflow (parse-only tools)

For WALTZ, PASTA, AggreProt, ArchCandy, Cross-Beta (no live runner yet), use
`amyloid-wrappers batch` for PATH+APPNN, then per-protein `amyloid-parse`:

```bash
amyloid-parse waltz --input raw/${ID}.dat --fasta fasta_split/${ID}.fasta \
  -o output_dir/waltz/parsed/${ID}_waltz.csv
amyloid-merge output_dir/*/parsed/${ID}_*.csv --fasta fasta_split/${ID}.fasta \
  -o output_dir/merged/${ID}_merged.csv
```

Or merge with BHT reference check: `amyloids-widemerge --reference …`.

---

## Development roadmap (v0.2.4)

| Phase | Status | Next steps |
| ----- | ------ | ---------- |
| **0** | done | 7 parsers, merge, cache, config, golden tests |
| **1** | done | PATH/APPNN runners, `batch`, `amyloids-widemerge`, live validation on `test.fasta` |
| **2** | planned | WALTZ local runner, `BaseWebRunner` for Selenium tools |
| **3** | planned | Web runners: AggreProt, ArchCandy, Cross-Beta, PASTA (`legacy/api/`) |
| **4** | planned | Aggrescan parser + runner |
| **5** | planned | `amyloid-metascore` CLI, CI on GitHub Actions |

**Validated (Phase 1):** `python -m amyloid_wrappers batch vendor/PATH/test.fasta -o output_dir`
with Modeller + PyRosetta + R `appnn` in env `wrappers`.

---

## Relation to `BHT_amyloid/`


| Old file              | Replacement                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `arch_cross_...ipynb` | `amyloid-parse` / `legacy/` copy                                    |
| `path_converter.py`   | `legacy/path_converter.py`, `runners/path.py`, `predictors/path.py` |
| `appnn_converter.R`   | `legacy/appnn_converter.R`, `runners/appnn.py`                      |
| `all/*_all.csv`       | wide merge via `amyloid-merge`; validate with `amyloids-widemerge --reference` |


