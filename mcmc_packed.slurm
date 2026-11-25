#!/bin/bash
#SBATCH --job-name=MCMC_packed
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --hint=nomultithread
#SBATCH -o logs/out/packed_%j.out
#SBATCH -e logs/err/packed_%j.err
## Set your cluster's partition here, or pass --partition=... (submit_packed.sh forwards FL_PARTITION):
##SBATCH --partition=YOUR_PARTITION
## Email notifications (optional):
##SBATCH --mail-type=END --mail-user=you@example.com
#
# One packed worker fills every core of its node: it reads $SLURM_CPUS_ON_NODE and pulls tasks from a
# shared work queue ($QUEUE_DIR/pending.txt + an flock'd counter), keeping that many single-core `srun`
# steps in flight. Launch N workers via submit_packed.sh; they cooperatively drain one queue (more
# nodes => faster; a worker that finds it empty just exits). --exclusive + --mem=0 give each worker the
# whole node; export PACK_CAP to cap concurrent tasks/node if memory-bound.

set -eE -o pipefail

# ---------- shared work queue (exported by submit_packed.sh) ----------
: "${QUEUE_DIR:?QUEUE_DIR not set — launch via submit_packed.sh, not bare sbatch}"
PENDING="$QUEUE_DIR/pending.txt"
CTR="$QUEUE_DIR/counter"
LOCK="$QUEUE_DIR/lock"
N_JOBIDX="${N_JOBIDX:-19}"
N_PAIRS="${N_PAIRS:-$(wc -l < "$PENDING")}"

# This node's core count, optionally capped (PACK_CAP) if RAM ever forces fewer concurrent tasks.
P="${SLURM_CPUS_ON_NODE:-1}"
if [ -n "${PACK_CAP:-}" ] && [ "$PACK_CAP" -lt "$P" ]; then P="$PACK_CAP"; fi

# ---------- repo root (portable: prefer the submit dir, fall back to this script's location) ----------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [ ! -f "$REPO_ROOT/src/calibration/mcmc_caller_script.jl" ]; then
  echo "FATAL: REPO_ROOT ($REPO_ROOT) is not the repo (mcmc_caller_script.jl missing)." >&2
  exit 1
fi

: "${JULIA_PROJECT:=$REPO_ROOT}"
JULIA_PROJECT_DIR="$REPO_ROOT"; cd "$JULIA_PROJECT_DIR"

# Activate a conda env if FL_CONDA_ENV is set (supplies the PyCall/PyPlot Python + numpy/scipy/pandas/
# matplotlib/seaborn). Otherwise `julia` and such a Python must already be on PATH.
if [ -n "${FL_CONDA_ENV:-}" ]; then
  set +u
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then . "$HOME/miniconda3/etc/profile.d/conda.sh"
  else eval "$(conda shell.bash hook)"; fi
  conda activate "$FL_CONDA_ENV"; set -u
  export PYTHON="${PYTHON:-$CONDA_PREFIX/bin/python}"
  [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ] && \
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi

BASE_OUTDIR="${JULIA_PROJECT_DIR}/${FL_OUTROOT:-outputs/mcmc}"
mkdir -p logs/out logs/err "$BASE_OUTDIR"

export JULIA_PROJECT="$JULIA_PROJECT_DIR" MPLBACKEND=Agg JULIA_PKG_PRECOMPILE_AUTO=0
[ -n "${JULIA_DEPOT_PATH:-}" ] && export JULIA_DEPOT_PATH   # else Julia's default ~/.julia
# one BLAS/OMP thread per task so the P concurrent tasks don't oversubscribe the P cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1

echo "worker job=${SLURM_JOB_ID} node=${SLURM_NODELIST} P=${P} cores  queue=${QUEUE_DIR}  N_PAIRS=${N_PAIRS}  FL_SERIAL=${FL_SERIAL:-?}  FL_OUTROOT=${FL_OUTROOT:-outputs/mcmc}"

# ---------- atomic "claim next pending line" (1-based line number; empty when drained) ----------
# The counter holds how many lines have been handed out; the next line is that + 1 (pending.txt is
# 1-based, as `sed -n Np` expects). Returns "" once all N_PAIRS lines have been claimed.
claim_next() {
  exec 9>"$LOCK"; flock 9
  local c line=""; c=$(<"$CTR")
  if [ "$c" -lt "$N_PAIRS" ]; then line=$(( c + 1 )); echo "$line" > "$CTR"; fi
  flock -u 9; exec 9>&-
  printf '%s' "$line"
}

launched=0; inflight=0
while :; do
  i=$(claim_next)                                   # 1-based line index, or "" when queue empty
  [ -z "$i" ] && break
  read -r REAL JOBIDX < <(sed -n "${i}p" "$PENDING") # e.g. "37 12"
  [ -z "${REAL:-}" ] && continue

  OUTDIR="${BASE_OUTDIR}/output_R${REAL}_job${SLURM_JOB_ID}_task${JOBIDX}"
  mkdir -p "$OUTDIR"
  LOG_OUT="logs/out/mcmc_R${REAL}_job_${SLURM_JOB_ID}_task_${JOBIDX}.out"
  LOG_ERR="logs/err/mcmc_R${REAL}_job_${SLURM_JOB_ID}_task_${JOBIDX}.err"

  sleep $(( RANDOM % 3 )) >/dev/null 2>&1 || true   # tiny jitter to avoid lockstep FS hits

  # one core per task; loop vars reach the step via --export (cleaner than inline quoting).
  srun --exclusive --exact -N1 -n1 -c1 --cpu-bind=cores \
       --export=ALL,REAL="$REAL",JOBIDX="$JOBIDX",OUTDIR="$OUTDIR",JULIA_PROJECT_DIR="$JULIA_PROJECT_DIR" \
       --output="${LOG_OUT}" --error="${LOG_ERR}" \
       bash -c '
         set -euo pipefail
         SECONDS=0
         julia "$JULIA_PROJECT_DIR/src/calibration/mcmc_caller_script.jl" "$JOBIDX" "$REAL" "$OUTDIR" 2>&1 \
           | sed -E "/CHOLMOD version|SparseArrays.CHOLMOD|compiled with CHOLMOD|currently linked with version|sparse matrix factorizations|download the generic binaries/d"
         dur=$SECONDS
         printf "Total runtime (R%s job%s): %02dh:%02dm:%02ds (=%ds)\n" "$REAL" "$JOBIDX" $((dur/3600)) $(((dur%3600)/60)) $((dur%60)) "$dur"
       ' &

  launched=$(( launched + 1 )); inflight=$(( inflight + 1 ))
  if [ "$inflight" -ge "$P" ]; then wait -n; inflight=$(( inflight - 1 )); fi
done

wait
echo "worker ${SLURM_JOB_ID} on ${SLURM_NODELIST}: drained queue, ran ${launched} tasks (P=${P})."