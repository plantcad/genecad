# Sourced by predict.sh and train.sh; defines genecad_version.

# Print the GeneCAD version for the installation in directory $1, followed by
# the git commit when $1 is the root of a git checkout. Never fails: prints
# "unknown" when no version can be found.
genecad_version() {
    local root="$1" version="" commit=""
    if [[ -f "$root/pyproject.toml" ]]; then
        # Source checkout or Docker image
        version=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$root/pyproject.toml" | head -n 1)
    else
        # Installed wheel, which ships predict.sh but not pyproject.toml
        version=$("${GENECAD_PYTHON:-python3}" -c \
            'import importlib.metadata as m; print(m.version("genecad"))' 2>/dev/null) || version=""
    fi
    if [[ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null)" == "$(cd "$root" && pwd -P)" ]]; then
        commit=$(git -C "$root" describe --tags --always --dirty 2>/dev/null) || commit=""
    fi
    echo "${version:-unknown}${commit:+ (git $commit)}"
}
