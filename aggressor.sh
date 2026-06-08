#!/usr/bin/env bash
#
# Install the AGGRESSOR command-line tool.
#
# Install strategy:
#   1. PRIMARY  : download the package (aggressor.toml + aggressor/ package)
#                 from GitHub, build a wheel, install it, then delete every
#                 downloaded/temporary file.
#   2. FALLBACK : if the GitHub step fails (no network, bad ref, ...), build
#                 and install from the local files next to this script. In
#                 this case ALL local files are preserved and the built wheel
#                 is kept in a versioned folder  aggressor_<version>/.
#
# Usage:
#   ./aggressor.sh                                  # GitHub, fallback to local
#   AGGRESSOR_LOCAL=1 ./aggressor.sh                # force local install
#   AGGRESSOR_REPO=https://github.com/u/r.git ./aggressor.sh
#   AGGRESSOR_REF=v1.0.0 ./aggressor.sh             # branch / tag
#   PYTHON_BIN=python3.11 ./aggressor.sh
#
# After installation the tool is available as:
#   aggressor --help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# GitHub source (override with env vars). Point AGGRESSOR_REPO at your fork.
AGGRESSOR_REPO="${AGGRESSOR_REPO:-https://github.com/georgeegreat/BHT_amyloid.git}"
AGGRESSOR_REF="${AGGRESSOR_REF:-main}"

# --------------------------------------------------------------------------
# Read the single source-of-truth version from a checkout's __init__.py.
# Arg: $1 = directory containing aggressor/__init__.py
# --------------------------------------------------------------------------
read_version() {
    "${PYTHON_BIN}" - "$1/aggressor/__init__.py" <<'PY'
import ast, sys
for node in ast.walk(ast.parse(open(sys.argv[1]).read())):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if getattr(target, "id", None) == "__version__":
                print(node.value.value)
                sys.exit(0)
sys.exit("__version__ not found in __init__.py")
PY
}

# --------------------------------------------------------------------------
# Build a wheel from a source directory (which must contain aggressor.toml and
# the aggressor/ package) and install it.
# Args: $1 = source dir, $2 = keep flag (1 -> persist aggressor_<version>/)
# --------------------------------------------------------------------------
build_and_install() {
    local src="$1" keep="$2"

    local version
    version="$(read_version "${src}")" || return 1

    # Stage the build inputs, mapping aggressor.toml -> pyproject.toml (the name
    # the build backend expects). Building from a staging copy keeps the source
    # tree clean (no build/ or *.egg-info left behind).
    local stage
    stage="$(mktemp -d)"
    mkdir -p "${stage}/pkg"
    cp -r "${src}/aggressor" "${stage}/pkg/aggressor"
    cp "${src}/aggressor.toml" "${stage}/pkg/pyproject.toml"

    local out
    if [ "${keep}" = "1" ]; then
        out="${SCRIPT_DIR}/aggressor_${version}"
        rm -rf "${out}"
    else
        out="${stage}/wheel"
    fi

    echo "Building AGGRESSOR ${version} (wheel dir: ${out})"
    if ! "${PYTHON_BIN}" -m pip wheel "${stage}/pkg" --no-deps --wheel-dir "${out}"; then
        rm -rf "${stage}"
        return 1
    fi
    if ! "${PYTHON_BIN}" -m pip install --force-reinstall "${out}"/aggressor-*.whl; then
        rm -rf "${stage}"
        return 1
    fi

    rm -rf "${stage}"
}

# --------------------------------------------------------------------------
# Download the package from GitHub, install it, then remove all downloaded
# files. Returns non-zero on any failure so the caller can fall back to local.
# --------------------------------------------------------------------------
install_from_github() {
    echo "Attempting GitHub install: ${AGGRESSOR_REPO} (ref: ${AGGRESSOR_REF})"
    local tmp
    tmp="$(mktemp -d)"
    local ok=0

    if command -v git >/dev/null 2>&1; then
        if git clone --depth 1 --branch "${AGGRESSOR_REF}" \
                "${AGGRESSOR_REPO}" "${tmp}/src"; then
            ok=1
        fi
    elif command -v curl >/dev/null 2>&1; then
        local url="${AGGRESSOR_REPO%.git}/archive/refs/heads/${AGGRESSOR_REF}.tar.gz"
        echo "git not available; downloading tarball: ${url}"
        mkdir -p "${tmp}/src"
        if curl -fsSL "${url}" -o "${tmp}/src.tgz" \
                && tar -xzf "${tmp}/src.tgz" -C "${tmp}/src" --strip-components=1; then
            ok=1
        fi
    else
        echo "Neither git nor curl is available for GitHub installation."
    fi

    if [ "${ok}" != "1" ] || [ ! -f "${tmp}/src/aggressor.toml" ]; then
        rm -rf "${tmp}"
        return 1
    fi

    if build_and_install "${tmp}/src" 0; then
        rm -rf "${tmp}"
        echo "GitHub install complete; downloaded files removed."
        return 0
    fi

    rm -rf "${tmp}"
    return 1
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
echo "Using interpreter: $(${PYTHON_BIN} --version 2>&1) (${PYTHON_BIN})"
"${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null 2>&1 || true

if [ "${AGGRESSOR_LOCAL:-0}" = "1" ]; then
    echo "Installing from local files in ${SCRIPT_DIR} (all files preserved)."
    build_and_install "${SCRIPT_DIR}" 1 || { echo "Local installation failed."; exit 1; }
elif install_from_github; then
    :
else
    echo "GitHub installation failed; falling back to local files in ${SCRIPT_DIR}."
    build_and_install "${SCRIPT_DIR}" 1 || { echo "Local installation failed."; exit 1; }
fi

echo
echo "Installation complete. Verify with:"
echo "    aggressor --help"
