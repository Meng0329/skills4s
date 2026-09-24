#!/bin/bash
# exp13 relaunch watcher — waits for GPU0 to free, then runs the FULL EXP13 flow.
# Trigger: GPU0 free memory >= 30GiB (DPO's 28GB block gone) AND no other exp13 run alive.
LOG=/data/mzb/skills4s/outputs/exp13_watcher.log
RUNLOG=/data/mzb/skills4s/outputs/exp13_run.log
cd /data/mzb/skills4s/experiments/exp13_contextual_write_remap
for i in $(seq 1 720); do
  if pgrep -f "exp13_contextual_write_remap/run.py" >/dev/null; then
    echo "$(date +%FT%T) exp13 run already alive; watcher exits" >> "$LOG"
    exit 0
  fi
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  echo "$(date +%FT%T) check#$i free=${FREE}MiB util=${UTIL}%" >> "$LOG"
  if [ "$FREE" -ge 30000 ]; then
    echo "$(date +%FT%T) GPU0 free=${FREE}MiB; launching full EXP13 run" >> "$LOG"
    nohup env CUDA_VISIBLE_DEVICES=0 python run.py > "$RUNLOG" 2>&1 &
    echo "$(date +%FT%T) LAUNCHED pid=$!" >> "$LOG"
    exit 0
  fi
  sleep 300
done
echo "$(date +%FT%T) watcher timed out after 60h without free GPU0" >> "$LOG"
