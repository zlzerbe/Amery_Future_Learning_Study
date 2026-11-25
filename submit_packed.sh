#!/bin/bash
set -euo pipefail
#
# submit_packed.sh <A|B|C|D> — run all 100×19 = 1900 mcmc calibrations of ONE configuration, filling every
# core of every node it is given. Each worker pulls from a shared work queue sized to its own
# $SLURM_CPUS_ON_NODE (see mcmc_packed.slurm), so heterogeneous node sizes and cheap-vs-expensive
# tasks balance out with no fixed per-node packing.
#
# Prereq: the configuration's future observations must already exist (see src/future_obs/AmeryFutureObservations.jl).
# Usage:
#   bash submit_packed.sh D                 # main configuration (Wiener + code-off)
#   NJOBS=8 bash submit_packed.sh A         # use fewer nodes
#   RERUN=1 bash submit_packed.sh C         # only (REAL,JOBIDX) whose GQs output is still missing
#
# Cluster env: FL_PARTITION (sbatch partition), FL_CONDA_ENV (conda env with the Python deps) — see README.
# Knobs (env): NJOBS (# independent 1-node workers, default 16), MAX_PARALLEL (concurrent workers,
# default = NJOBS), RERUN (1 = enqueue only missing outputs), PACK_CAP (cap tasks/node if RAM-bound).

EXPT="${1:?usage: submit_packed.sh <A|B|C|D>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- geometry ----
N_REAL=100          # realizations 1..100
N_JOBIDX=19         # job_idx 0..18
NJOBS="${NJOBS:-16}"                      # independent 1-node workers to submit (queue self-drains)
MAX_PARALLEL="${MAX_PARALLEL:-$NJOBS}"    # cap concurrently-running workers
RERUN="${RERUN:-0}"

# ---- predict_f everywhere + per-configuration error-model / code-noise flags ----
export FL_GP_PREDICT=f
case "$EXPT" in
  A) export FL_ERR_MODEL=indep  FL_CODE_NOISE=1 FL_SERIAL=seed9_indep_code    FL_OUTROOT=outputs/mcmc_indep_code    FL_TARGET_ACCEPT=0.8  ;;
  B) export FL_ERR_MODEL=wiener FL_CODE_NOISE=1 FL_SERIAL=seed9_wiener_code   FL_OUTROOT=outputs/mcmc_wiener_code   FL_TARGET_ACCEPT=0.8  ;;
  C) export FL_ERR_MODEL=indep  FL_CODE_NOISE=0 FL_SERIAL=seed9_indep_nocode  FL_OUTROOT=outputs/mcmc_indep_nocode  FL_TARGET_ACCEPT=0.95 ;;
  D) export FL_ERR_MODEL=wiener FL_CODE_NOISE=0 FL_SERIAL=seed9_wiener_nocode FL_OUTROOT=outputs/mcmc_wiener_nocode FL_TARGET_ACCEPT=0.95 ;;
  *) echo "unknown experiment '$EXPT' (use A, B, C, or D)"; exit 1 ;;
esac

# Force the PRODUCTION sampling knobs (never a smoke-test override lingering in the shell).
unset FL_BURNIN FL_N_ADAPTS FL_N_POST || true

obs="intermediate_data/future_observations/${FL_SERIAL}_future_observations.jld2"
if [ ! -f "$obs" ]; then
  echo "ERROR: $obs missing. Generate the future observations for configuration $EXPT first (src/future_obs/AmeryFutureObservations.jl)." >&2; exit 1
fi

# Slurm opens the worker -o/-e files at submit time and won't create parent dirs.
mkdir -p logs/out logs/err "$FL_OUTROOT"

# ---- build the shared work queue: one "REAL JOBIDX" line per task ----
# RERUN=1 skips pairs whose completion marker <OUTDIR>/gq_dir/<lastyear>_GQs.jld2 already exists
# (lastyear = 2030 + JOBIDX*15), matched across any prior job-id via a glob — so a re-submit only
# fills the gaps left by dead/killed workers instead of redoing all 1900.
QUEUE_DIR="$REPO_ROOT/logs/queue_${EXPT}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$QUEUE_DIR"
PENDING="$QUEUE_DIR/pending.txt"; : > "$PENDING"
skipped=0
for (( R=1; R<=N_REAL; R++ )); do
  for (( J=0; J<N_JOBIDX; J++ )); do
    if [ "$RERUN" = "1" ]; then
      lastyear=$(( 2030 + J * 15 ))
      # any completed output for this (R,J) regardless of the job-id in the dir name
      if compgen -G "$FL_OUTROOT/output_R${R}_"*"_task${J}/gq_dir/${lastyear}_GQs.jld2" > /dev/null; then
        skipped=$(( skipped + 1 )); continue
      fi
    fi
    printf '%s %s\n' "$R" "$J" >> "$PENDING"
  done
done
echo 0 > "$QUEUE_DIR/counter"
N_PAIRS=$(wc -l < "$PENDING")
if [ "$N_PAIRS" -eq 0 ]; then
  echo "nothing to do: all outputs for configuration $EXPT already present (skipped $skipped)."; rm -rf "$QUEUE_DIR"; exit 0
fi
export QUEUE_DIR N_PAIRS N_JOBIDX

echo "configuration $EXPT: FL_SERIAL=$FL_SERIAL  FL_OUTROOT=$FL_OUTROOT  FL_GP_PREDICT=$FL_GP_PREDICT"
echo "queue=$QUEUE_DIR  N_PAIRS=$N_PAIRS pending$( [ "$RERUN" = 1 ] && echo " (skipped $skipped already-done)" )  workers=$NJOBS (≤$MAX_PARALLEL concurrent)"

# ---- fan out NJOBS independent 1-node workers, throttled to MAX_PARALLEL concurrent ----
active_count() { squeue -u "$USER" -h -n MCMC_packed 2>/dev/null | wc -l; }
for (( n=1; n<=NJOBS; n++ )); do
  while [ "$(active_count)" -ge "$MAX_PARALLEL" ]; do sleep 10; done
  sbatch ${FL_PARTITION:+--partition="$FL_PARTITION"} --export=ALL "$REPO_ROOT/mcmc_packed.slurm"
done
echo "submitted $NJOBS workers; they cooperatively drain $QUEUE_DIR (scheduler keeps ≤$MAX_PARALLEL running)."