#!/bin/bash
# Post-merge cleanup of scene-shard worker intermediates (orbit-dev, 2026-09-01).
#
# Called by shard_rollout.sh after a successful merge when CLEANUP_SHARDS=1
# (CLEANUP_SHARDS=dry prints the plan without deleting). For each shard i of P:
#   * the three suffixed worker FILES <out>-shard{i}of{P}.<ext> (exact paths,
#     derived the same way the driver derives them for merging);
#   * the driver's per-shard stdout   <dir(OUT_JSON)>/shard{i}.stdout;
#   * worker output DIRS matching the exact name pattern *-shard{i}of{P},
#     searched ONLY in dir(OUT_JSON) and its parent (the two places the
#     suffixed --output-dir convention can land them) -- and deleted ONLY if
#     they contain NO *.hdf5 anywhere inside (data safety: hdf5 inside means
#     someone pointed real outputs there; such a dir is left untouched and
#     reported).
# The merged outputs (OUT_JSON/OUT_SUCCESS/OUT_ALL) are never touched.
#
# Usage: _shard_cleanup.sh P OUT_JSON OUT_SUCCESS OUT_ALL [dry]
set -u
P=$1; OUT_JSON=$2; OUT_SUCCESS=$3; OUT_ALL=$4
DRY=${5:-}

say() { echo "[shard_cleanup] $*"; }
rm_maybe() {  # rm_maybe <path> <kind>
  if [ -e "$1" ]; then
    if [ "$DRY" = "dry" ]; then
      say "DRY would remove $2: $1"
    else
      rm -rf -- "$1" && say "removed $2: $1"
    fi
  fi
}

BASE_J=$(dirname "$OUT_JSON")
removed_files=0; kept_dirs=0

for i in $(seq 0 $((P - 1))); do
  suf="-shard${i}of${P}"
  for out in "$OUT_JSON" "$OUT_SUCCESS" "$OUT_ALL"; do
    low="${out,,}"
    if [[ "$low" == *.hdf5 || "$low" == *.json ]]; then
      p="${out%.*}${suf}.${out##*.}"
    else
      p="${out}${suf}"
    fi
    if [ -e "$p" ]; then
      rm_maybe "$p" "worker file"
      removed_files=$((removed_files + 1))
    fi
  done
  rm_maybe "$BASE_J/shard$i.stdout" "worker stdout"
  # guarded worker-dir removal (never if any hdf5 inside)
  for d in "$BASE_J" "$BASE_J/.."; do
    for cand in "$d"/*"$suf"; do
      [ -d "$cand" ] || continue
      if find "$cand" -name "*.hdf5" -print -quit | grep -q .; then
        say "KEPT (contains hdf5 -- not worker residue?): $cand"
        kept_dirs=$((kept_dirs + 1))
      else
        rm_maybe "$cand" "worker dir"
      fi
    done
  done
done

# absolute safety net: the merged outputs must still exist
for out in "$OUT_JSON" "$OUT_SUCCESS" "$OUT_ALL"; do
  if [ ! -e "$out" ]; then
    say "WARNING: merged output missing after cleanup: $out"
  fi
done
say "done (${DRY:+DRY }intermediates cleared, kept-dirs=$kept_dirs)"
