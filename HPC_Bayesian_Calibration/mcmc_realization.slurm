#!/bin/bash
#SBATCH --job-name=MCMC_N_All
#SBATCH --partition=cpu,p100
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=19
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=36:00:00
#SBATCH --hint=nomultithread
#SBATCH -o slurm_output_new_all/launcher_%j.out
#SBATCH -e slurm_errors_new_all/launcher_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=sjantre@bnl.gov

# ---------- strict mode (delay nounset until env is ready) ----------
set -eE -o pipefail

# Read REALIZATION_ID from env or first arg; default to 1
REALIZATION_ID="${REALIZATION_ID:-${1:-1}}"

# Predefine vars some hooks expect so nounset won’t choke later
: "${JULIA_DEPOT_PATH:=$HOME/.julia_depot_1.10}"
: "${JULIA_PROJECT:=/home/sjantre/Projects/Future_Learning/BrookhavenCode}"

# ---- conda init (nounset OFF while sourcing/activating) ----
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  set +u
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate julia_new
  set -u
else
  export PATH="$HOME/miniconda3/bin:$PATH"
  set +u
  eval "$(conda shell.bash hook)"
  conda activate julia_new
  set -u
fi

# Now safe to enable nounset
set -u

# ---------- paths & env ----------
JULIA_PROJECT_DIR="/home/sjantre/Projects/Future_Learning/BrookhavenCode"
mkdir -p slurm_output_new_all slurm_errors_new_all "${JULIA_PROJECT_DIR}/MCMC_Outputs_Sanket_Failed_ReRuns_New_All"

export JULIA_DEPOT_PATH
export JULIA_PROJECT="$JULIA_PROJECT_DIR"
export MPLBACKEND=Agg
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1"
export PYTHON="${CONDA_PREFIX}/bin/python"
export JULIA_PKG_PRECOMPILE_AUTO=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1

# ---------- warm up once (avoid 19 concurrent Pkg ops) ----------
julia -e 'using Pkg; Pkg.activate(ENV["JULIA_PROJECT"]); Pkg.instantiate(); Pkg.precompile();
          ENV["PYTHON"]=ENV["CONDA_PREFIX"]*"/bin/python";
          Pkg.add(["PyCall","PyPlot","StatsPlots"]); Pkg.build("PyCall")' \
  |& grep -v -E "CHOLMOD version|SparseArrays\.CHOLMOD|compiled with CHOLMOD|currently linked with version|sparse matrix factorizations|download the generic binaries"

# ---------- launch 19 parallel tasks on THIS node ----------
for TASK_IDX in $(seq 0 18); do
  OUTDIR="${JULIA_PROJECT_DIR}/MCMC_Outputs_Sanket_Failed_ReRuns_New_All/output_R${REALIZATION_ID}_job${SLURM_JOB_ID}_task${TASK_IDX}"
  mkdir -p "$OUTDIR"

  LOG_OUT="slurm_output_new_all/mcmc_R${REALIZATION_ID}_job_${SLURM_JOB_ID}_task_${TASK_IDX}.out"
  LOG_ERR="slurm_errors_new_all/mcmc_R${REALIZATION_ID}_job_${SLURM_JOB_ID}_task_${TASK_IDX}.err"

  # tiny jitter to avoid lockstep FS hits
  sleep $((RANDOM % 3)) >/dev/null 2>&1 || true

  srun --exclusive -N1 -n1 -c1 --cpu-bind=cores \
       --output="${LOG_OUT}" --error="${LOG_ERR}" \
       bash -c '
         set -euo pipefail
         SECONDS=0
         julia "'"$JULIA_PROJECT_DIR"'/mcmc_caller_script.jl" '"$TASK_IDX"' '"$REALIZATION_ID"' "'"$OUTDIR"'" \
           |& grep -v -E "CHOLMOD version|SparseArrays\.CHOLMOD|compiled with CHOLMOD|currently linked with version|sparse matrix factorizations|download the generic binaries"
         dur=$SECONDS
         printf "Total runtime (task %s): %02dh:%02dm:%02ds (=%ds)\n" "'"$TASK_IDX"'" $((dur/3600)) $(((dur%3600)/60)) $((dur%60)) "$dur"
       ' &
done

wait
echo "Realization ${REALIZATION_ID}: all 19 tasks finished on ${SLURM_NODELIST} (job ${SLURM_JOB_ID})."