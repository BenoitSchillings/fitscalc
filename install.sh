#!/usr/bin/env bash
# install.sh - add a `calc` shell function to ~/.bashrc that runs fitscalc.py.
#
# Re-running this script is safe; it replaces any prior fitscalc block, so it
# also fixes the path if you move the repo. Run with --uninstall to remove.
#
# Usage:
#   ./install.sh              install or refresh
#   ./install.sh --uninstall  remove

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${SCRIPT_DIR}/fitscalc.py"
RC="${HOME}/.bashrc"
BEGIN='# >>> fitscalc >>>'
END='# <<< fitscalc <<<'

if [[ ! -f "$TARGET" ]]; then
    echo "error: $TARGET not found" >&2
    exit 1
fi

remove_block() {
    if [[ -f "$RC" ]] && grep -qF "$BEGIN" "$RC"; then
        local tmp
        tmp="$(mktemp)"
        awk -v b="$BEGIN" -v e="$END" '
            $0==b {skip=1; next}
            $0==e {skip=0; next}
            !skip {print}
        ' "$RC" > "$tmp"
        mv "$tmp" "$RC"
        echo "removed existing fitscalc block from $RC"
    fi
}

if [[ "${1:-}" == "--uninstall" ]]; then
    remove_block
    echo "uninstalled. open a new shell for the change to take effect."
    exit 0
fi

remove_block

cat >> "$RC" <<EOF
$BEGIN
calc() {
    python3 "$TARGET" "\$@"
}
$END
EOF

echo "added \`calc\` function to $RC pointing at $TARGET"
echo "open a new shell, or run:  source \"$RC\""
