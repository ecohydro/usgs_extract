#!/bin/bash
#SBATCH --job-name=usgs_metadata_unlisted
#SBATCH -p grit_nodes
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append
#SBATCH --exclude=hpc-10.grit.ucsb.edu,hpc-13.grit.ucsb.edu,hpc-15.grit.ucsb.edu
#SBATCH --output=/home/annaboser/hpc_logs/usgs_metadata_unlisted_r%a_sbatch.log
# USGS metadata, unlisted pages (Anna, 2026-11-18): the 4,067 pages with a Reducto table but no table on the published
# list (shards 319-339, staged by 01_full_run.py prepare_unlisted). Same worker as the full run; when every shard is
# written, worker 0 gives unreadable answers the second reading (01_full_run.py reread, which skips pages reread before).
# Six identical workers share one list of shards (code/01_digitization/04_metadata_extraction/01_full_run.py worker): each claims
# a free shard, reads it, writes it, repeats, and exits when every shard is written. Resumable: resubmit at any time.
# Inputs and outputs are on the home share (data/analysis/evaluation/metadata/full_run); the model is on hpc-scratch.
echo "=== sbatch job $SLURM_JOB_ID worker $SLURM_ARRAY_TASK_ID on $(hostname) $(date)"
PY=/home/annaboser/miniconda3/envs/vllm-rs/bin/python
CODE=/home/annaboser/usgs_extract/code/01_digitization/04_metadata_extraction
MODEL=/home/hpc-scratch/annaboser/usgs_models/hf/hub
hop() {
  BAD="${BAD_NODES:+$BAD_NODES,}$SLURMD_NODENAME"
  echo "$1 ON $(hostname) (slurm node name $SLURMD_NODENAME): resubmitting with --exclude=$BAD"
  sbatch --array=$SLURM_ARRAY_TASK_ID --exclude="hpc-10.grit.ucsb.edu,hpc-13.grit.ucsb.edu,hpc-15.grit.ucsb.edu,$BAD" --export=ALL,BAD_NODES="$BAD" "$0" && exit 0
  echo "RESUBMISSION FAILED"; exit 1
}
timeout 60 ls "$MODEL" >/dev/null 2>&1 || hop "MODEL FOLDER NOT REACHABLE"
timeout 60 ls "$CODE/01_full_run.py" >/dev/null 2>&1 || hop "CODE NOT REACHABLE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || hop "NO GPU"
[ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -ge 40000 ] || hop "GPU TOO SMALL (needs the 46 GB L40S)"
# vLLM asks for 90% of the GPU; another job on a shared GPU (2026-11-19: 7.5 GB in use on hpc-03) makes it fail at start
[ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 41500 ] || hop "GPU MEMORY IN USE BY ANOTHER JOB"
timeout 600 $PY -c "import vllm, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 || hop "PYTHON ENV OR CUDA CANNOT LOAD"
# an older copy of the same worker already running means this one is a duplicate submission
if squeue -h -u annaboser -n "$SLURM_JOB_NAME" -t RUNNING -o "%A %K" 2>/dev/null | awk -v me="$SLURM_JOB_ID" -v k="$SLURM_ARRAY_TASK_ID" '($1 + 0) < (me + 0) && $2 == k {f=1} END {exit !f}'; then
  echo "AN OLDER $SLURM_JOB_NAME worker $SLURM_ARRAY_TASK_ID IS RUNNING: this copy exits $(date)"; exit 0
fi
cd "$CODE"
timeout 23h $PY 01_full_run.py worker
rc=$?
if [ $rc -eq 124 ]; then
  echo "WALL TIME: worker $SLURM_ARRAY_TASK_ID resubmitting $(date)"
  sbatch --array=$SLURM_ARRAY_TASK_ID ${BAD_NODES:+--exclude="hpc-10.grit.ucsb.edu,hpc-13.grit.ucsb.edu,hpc-15.grit.ucsb.edu,$BAD_NODES"} --export=ALL,BAD_NODES="${BAD_NODES:-}" "$0" || echo "RESUBMISSION FAILED $(date)"
else
  echo "=== worker $SLURM_ARRAY_TASK_ID exited rc=$rc $(date)"
  if [ $rc -eq 0 ] && [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    timeout 6h $PY 01_full_run.py reread && echo "=== REREAD DONE $(date)" || echo "=== REREAD FAILED rc=$? $(date)"
  fi
fi
