# AGGRESSOR

**A**ggregation-**G**uided **G**eneration of **RE**gion-specific **S**ubstitution-**OR**iented mutations.

AGGRESSOR is a rule-based *in silico* mutagenesis toolkit for protein sequences,
distributed as an installable Python package (the `aggressor/` package).
It detects aggregation-prone regions (APRs), then generates single- and
multi-point mutants designed to disrupt amyloidogenic potential, including
gatekeeper substitutions in the residues flanking each APR.

Results are fully reproducible: the output is independent of the number of
worker threads and of the Python hash seed.

---

## Installation

### Files required to install

Only these are needed to build and install the command-line tool:

| File / folder    | Purpose                                                          |
| ---------------- | ---------------------------------------------------------------- |
| `aggressor/`     | The Python package (all modules, see structure below).           |
| `aggressor.toml` | Build configuration; the version is read from `__init__.py`.     |
| `aggressor.sh`   | Installer: pulls from GitHub, falls back to local files.         |

> The build backend expects the configuration to be named `pyproject.toml`.
> `aggressor.sh` handles this automatically by staging `aggressor.toml` as
> `pyproject.toml` in a temporary build directory, so the source tree keeps the
> `aggressor.toml` name and stays clean.

### Recommended — install script

```bash
./aggressor.sh
```

The installer works in two stages:

1. **GitHub (primary):** downloads the package (`aggressor.toml` + the
   `aggressor/` package) from GitHub, builds a wheel, installs it, and then
   **deletes every downloaded/temporary file**.
2. **Local (fallback):** if the GitHub step fails (no network, bad ref, …), it
   builds and installs from the local files next to the script. In this case
   **all local files are preserved** and the built wheel is kept in a versioned
   folder `aggressor_<version>/` (e.g. `aggressor_1.0.0/`).

Useful environment variables:

```bash
AGGRESSOR_REPO=https://github.com/<you>/aggressor.git ./aggressor.sh  # source repo
AGGRESSOR_REF=v1.0.0 ./aggressor.sh                                   # branch / tag
AGGRESSOR_LOCAL=1 ./aggressor.sh                                      # force local install
PYTHON_BIN=python3.11 ./aggressor.sh                                  # choose the interpreter
```

After installation the tool is available directly on the command line:

```bash
aggressor --help
```

### Running without installing

```bash
python -m aggressor protein.fasta [options]      # run the package in-place
python aggressor/main.py protein.fasta [options] # run the package bootstrap
```

---

## Package structure

```
aggressor/
├── __init__.py      # package version (single source of truth) + public API
├── __main__.py      # enables `python -m aggressor`
├── main.py          # bootstrap for `python aggressor/main.py`
├── cli.py           # argument parsing, validation, pipeline orchestration
├── config.py        # constants, amino-acid scales, logging setup
├── models.py        # data structures (clusters, mutation types, results)
├── rules.py         # aggregation rule evaluators
├── clustering.py    # Union-Find cluster merging
├── analysis.py      # region analysis + mutation-type classification
├── mutagenesis.py   # single- and multi-point mutation generation
├── sequtils.py      # sequence validation + region parsing
├── fasta_io.py      # FASTA reading/writing + output directory layout
└── reporting.py     # help text and human-readable summaries
```

---

## Generated paths and folders

| Path                       | When                              | Contents                                                          |
| -------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| `aggressor_<version>/`     | local install (`aggressor.sh`)    | The built wheel (e.g. `aggressor-1.0.0-py3-none-any.whl`). Not created for a GitHub install (temporary files are removed). |
| `mutated_sequences.fasta`  | single-mutation mode              | All single-point mutants (override with `-o/--output`).           |
| `mutated_sequences/`       | `--multi-mutations` mode          | Multi-point mutants, organized by level and category (see below). |

Multi-mutation output layout (`--multi-output`, default `mutated_sequences/`):

```
mutated_sequences/
├── double_mutations/
│   ├── single_region.fasta
│   ├── multi_region.fasta
│   ├── all_gatekeeper.fasta
│   ├── all_core.fasta
│   └── mixed.fasta
└── triple_mutations/
    └── ...
```

---

## Quick start

```bash
# Rule-based mutagenesis over a region
aggressor protein.fasta --regions 60:95

# Whole-sequence scan
aggressor protein.fasta --regions all

# Tune the gatekeeper zone and include internal merge-zone gatekeepers
aggressor protein.fasta --regions 60:95 --gatekeeper-distance 3 --internal-gatekeepers

# Double and triple mutants with parallel workers
aggressor protein.fasta --regions 60:95 --multi-mutations 2 3 --threads 4
```

### Gatekeeper options

- `--gatekeeper-distance N` — number of residues flanking each APR (on each
  side) that are also mutated as part of the gatekeeper zone (default `3`;
  `0` restricts mutations to cluster core positions only). Flanking positions
  receive **only** gatekeeping substitutions (`-g`/`--gatekeeping`).
- `--internal-gatekeepers` — also mutate the internal gap residues inside
  merged clusters (the short zones between merged sub-clusters) as gatekeepers
  (default: off).

Run `aggressor --help` for the full option list.
