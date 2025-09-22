#!/bin/bash
set -euo pipefail

JOBFILE="mcmc_realization.slurm"
MAX_PARALLEL=16                 # at most 16 nodes/jobs active
USER_NAME="${USER}"

active_count() {
  # Count this job family (exact job name set in the slurm script: MCMC_N_All)
  squeue -u "$USER_NAME" -h -n MCMC_N_All | wc -l
}

# RIDS_failed = 4 15 20 25 44 47 67 81 82 89 100

# for RID in "${RIDS_failed[@]}"; do
for RID in {1..100}; do
  # throttle so only up to 10 jobs are active (PD+R)
  while [ "$(active_count)" -ge "$MAX_PARALLEL" ]; do
    sleep 10
  done

  # submit one realization; choose ONE of the two styles:

  # 1) pass as env var (preferred)
  sbatch --export=ALL,REALIZATION_ID="$RID" "$JOBFILE"

  # 2) or pass as positional arg (uncomment this and comment the line above)
  # sbatch "$JOBFILE" "$RID"
done

# echo "Submitted realizations 1..100; scheduler will keep up to $MAX_PARALLEL running concurrently."
echo "Submitted failed realizations: ${RIDS_failed[*]}; scheduler will keep up to $MAX_PARALLEL running concurrently."