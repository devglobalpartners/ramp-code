#!/usr/bin/env bash
#
# Usage: ./batch_run_predictions_on_geotiffs.sh \
#          -i INPUT_FOLDER_PATH \
#          -o OUTPUT_FOLDER_PATH \
#          -m MODEL_PATH \
#          [-s START_STEP]   # 1=tile (default), 2=predict, 3=geojson
#
set -euo pipefail

# ── Cleanup trap: kill orphan background jobs on exit ────────────────────────
cleanup() {
    local exit_code=$?
    jobs -p 2>/dev/null | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    if (( exit_code != 0 )); then
        echo ""
        echo "ERROR: Script exited with code $exit_code. Intermediate files preserved for debugging."
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# ── Parse arguments ──────────────────────────────────────────────────────────
start_step=1
while getopts i:o:m:s: flag; do
    case "${flag}" in
        i) input="${OPTARG}" ;;
        o) output="${OPTARG}" ;;
        m) model_raw="${OPTARG}" ;;
        s) start_step="${OPTARG}" ;;
        *) echo "Unknown flag: ${flag}"; exit 1 ;;
    esac
done

# Validate required args
: "${input:?  -i INPUT_FOLDER_PATH is required}"
: "${output:? -o OUTPUT_FOLDER_PATH is required}"
: "${model_raw:?  -m MODEL_PATH is required}"

# Validate start_step
if [[ ! "$start_step" =~ ^[123]$ ]]; then
    echo "ERROR: -s must be 1, 2, or 3 (got: $start_step)"
    exit 1
fi

# Verify model file exists
if ! model=$(realpath -e "$model_raw" 2>/dev/null); then
    echo "ERROR: Model file not found: $model_raw"
    exit 1
fi

# Resolve script paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(realpath "$SCRIPT_DIR/../python")"
TILE_SCRIPT="$PYTHON_DIR/tile_dataset_1.py"
PRED_SCRIPT="$PYTHON_DIR/get_model_predictions.py"
LABEL_SCRIPT="${RAMP_HOME:-/tf}/scripts/get_labels_from_masks.py"

for script in "$TILE_SCRIPT" "$PRED_SCRIPT" "$LABEL_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "ERROR: Required script not found: $script"
        exit 1
    fi
done

echo "Input  : $input"
echo "Output : $output"
echo "Model  : $model"
echo "Start  : step $start_step"

mkdir -p "$output"/chips "$output"/pred_masks "$output"/geojson

# Log file
LOG="$output/run.log"
echo "Full log: $LOG"
echo "Resumed at step $start_step: $(date)" >> "$LOG"

# ── Helper: count files in a directory ───────────────────────────────────────
count_files() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -maxdepth 1 -type f | wc -l
    else
        echo 0
    fi
}

# ── Step 1: Tile all GeoTIFFs ────────────────────────────────────────────────
if (( start_step <= 1 )); then
    echo ""
    echo "==> Step 1: Tiling GeoTIFFs..."
    echo "[$(date +%H:%M:%S)] Tiling started" >> "$LOG"

    shopt -s globstar nullglob

    tif_files=()
    for file in "$input"/**/*.tif; do
        [[ -f "$file" ]] || continue
        tif_files+=("$file")
    done

    total=${#tif_files[@]}

    if (( total == 0 )); then
        echo "ERROR: No .tif files found in $input"
        exit 1
    fi
    echo "    Total TIFs: $total"

    done_count=0
    START_TIME=$(date +%s)
    BATCH_SIZE="${RAMP_TILE_WORKERS:-21}"   # 75% of 28 cores (configurable via env var)
    PIDS=()
    PID_FILES=()
    FAILED=0

    wait_for_batch() {
        local fail_in_batch=0
        for i in "${!PIDS[@]}"; do
            local pid="${PIDS[$i]}"
            local src="${PID_FILES[$i]}"
            if ! wait "$pid" 2>/dev/null; then
                echo "  WARNING: Tiling FAILED for: $src (PID $pid)" | tee -a "$LOG"
                (( ++fail_in_batch )) || true
            fi
        done
        FAILED=$(( FAILED + fail_in_batch ))
        PIDS=()
        PID_FILES=()
        return 0
    }

    for file in "${tif_files[@]}"; do
        (( ++done_count )) || true

        ELAPSED=$(( $(date +%s) - START_TIME ))
        if (( ELAPSED > 0 && done_count > 1 )); then
            ETA=$(( ELAPSED * (total - done_count) / done_count ))
            echo "  [$done_count/$total] $(basename "$file") | elapsed: ${ELAPSED}s | ETA: ~${ETA}s"
        else
            echo "  [$done_count/$total] $(basename "$file")"
        fi

        python "$TILE_SCRIPT" \
            -img "$file" \
            -out "$output" \
            -pfx tmp_chip \
            -ndt 0.4 \
            -oht 256 \
            -owd 256 >> "$LOG" 2>&1 &

        PIDS+=($!)
        PID_FILES+=("$file")

        if (( ${#PIDS[@]} >= BATCH_SIZE )); then
            CHIPS=$(count_files "$output/chips")
            echo "  Waiting for batch of ${#PIDS[@]}... (chips so far: $CHIPS)"
            wait_for_batch
            CHIPS=$(count_files "$output/chips")
            echo "  Batch done. Chips so far: $CHIPS"
        fi
    done

    if (( ${#PIDS[@]} > 0 )); then
        echo "  Waiting for final batch of ${#PIDS[@]}..."
        wait_for_batch
    fi

    TOTAL_TIME=$(( $(date +%s) - START_TIME ))
    CHIPS=$(count_files "$output/chips")
    echo ""
    echo "==> Tiling complete in ${TOTAL_TIME}s | Total chips: $CHIPS | Failed: $FAILED"
    echo "[$(date +%H:%M:%S)] Tiling complete — ${TOTAL_TIME}s — $CHIPS chips — $FAILED failures" >> "$LOG"

    if (( CHIPS == 0 )); then
        echo "ERROR: Zero chips produced. Cannot continue. Check $LOG for details."
        exit 1
    fi

    if (( FAILED > 0 )); then
        echo "WARNING: $FAILED/$total tiling jobs failed. Continuing with $CHIPS chips."
    fi
else
    CHIPS=$(count_files "$output/chips")
    echo ""
    echo "==> Step 1: SKIPPED (using $CHIPS existing chips in $output/chips)"
    if (( CHIPS == 0 )); then
        echo "ERROR: -s $start_step but $output/chips is empty. Nothing to work with."
        exit 1
    fi
fi

# ── Step 2: Run model predictions ────────────────────────────────────────────
if (( start_step <= 2 )); then
    echo ""
    echo "==> Step 2: Running building footprint predictions..."
    echo "[$(date +%H:%M:%S)] Predictions started" >> "$LOG"

    python "$PRED_SCRIPT" \
        -mod "$model" \
        -idir "$output"/chips \
        -odir "$output"/pred_masks >> "$LOG" 2>&1

    MASKS=$(count_files "$output/pred_masks")
    echo "==> Predictions complete. Masks produced: $MASKS"
    echo "[$(date +%H:%M:%S)] Predictions complete — $MASKS masks" >> "$LOG"

    if (( MASKS == 0 )); then
        echo "ERROR: Zero prediction masks produced. Check $LOG for details."
        exit 1
    fi
else
    MASKS=$(count_files "$output/pred_masks")
    echo ""
    echo "==> Step 2: SKIPPED (using $MASKS existing masks in $output/pred_masks)"
    if (( MASKS == 0 )); then
        echo "ERROR: -s $start_step but $output/pred_masks is empty. Nothing to work with."
        exit 1
    fi
fi

# ── Step 3: Convert masks to GeoJSON ─────────────────────────────────────────
echo ""
echo "==> Step 3: Converting masks to GeoJSON..."
echo "[$(date +%H:%M:%S)] GeoJSON conversion started" >> "$LOG"

python "$LABEL_SCRIPT" \
    -in  "$output"/pred_masks \
    -out "$output"/geojson/predicted_buildings.geojson \
    -bwpix 2 \
    -crs EPSG:4326 >> "$LOG" 2>&1

echo "==> GeoJSON complete."
echo "[$(date +%H:%M:%S)] GeoJSON complete" >> "$LOG"

# ── Step 4: Cleanup ──────────────────────────────────────────────────────────
echo ""
echo "==> Cleaning up temporary files..."
rm -rf "$output"/chips
rm -rf "$output"/pred_masks

echo ""
echo "Done! GeoJSON output: $output/geojson/predicted_buildings.geojson"
echo "Full log available at: $LOG"
echo "[$(date +%H:%M:%S)] All done." >> "$LOG"