#!/bin/bash
# rebuild_inventory.sh <catalog dir> <registry dir>: rebuild $IX_INVENTORY (inventory.jsonl) and
# $IX_RECIPE_RECON from the fetched inputs. With blis-catalog a2a8e47 and blis-registry 2e7070c (the
# coverage-branch heads the inventory was built at) and the inputs below, the result is sha256
# 678f244a4fea9b1bb5e29dc076a027d9840cd2a6378f62e5374db092020162bb (verified locally, 2026-10-17).
# Inputs: fetch/db_dump.sh, fetch/db_export.sh, fetch/gh_bmk.sh, fetch/recipes.sh inputs/nolog_shas.txt.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); I=$here/../inputs
DATA=${IX_DATA_DIR:-./ix-data}; DB=$DATA/db
INV=${IX_INVENTORY:-$DATA/inventory.jsonl}; RECON=${IX_RECIPE_RECON:-$DATA/recipe_recon.jsonl}
PY="uv run --no-project --with pyyaml python -I"
common=(--rows "$DB/rows.tsv" --logs "$DB/logs.tsv" --logs "$DB/logs2.tsv" --bmk-dir "$DATA/gh-bmk-postdump"
        --binding "$I/dsv41_binding_ids.json" --catalog "$1" --registry "$2" --extra "$I/inventory_extra.jsonl")
# pass 1 (no recipe reconstruction) -> the g_nolog runs reconstruct.py needs
$PY "$here/build_inventory.py" "${common[@]}" --out "$INV.pass1"
$PY "$here/reconstruct.py" --inventory "$INV.pass1" --recipes "${IX_RECIPES:-$DATA/recipes}" --heads "$DB/heads.tsv" --out "$RECON"
$PY "$here/build_inventory.py" "${common[@]}" --recipe-recon "$RECON" --out "$INV"
rm -f "$INV.pass1"
shasum -a 256 "$INV" "$RECON"
