#!/bin/bash
# ==============================================================================
# submit_parallel.sh — Submit parallel GPU jobs on PARAM Brahma (PBS)
#
# Splits kernel files for a group across multiple GPU jobs.
# ALL jobs run the SAME deviation — no independent deviation iteration.
#
# Usage:
#   ./submit_parallel.sh GROUP TOTAL_KERNELS NUM_GPUS DEVIATION
#   ./submit_parallel.sh 38 100 4 2    # group 38, 100 kernels, 4 GPUs, deviation 2
#   ./submit_parallel.sh 39 100 2 3    # group 39, 100 kernels, 2 GPUs, deviation 3
#
# What it does (example: 38, 100, 4, 2):
#   GPU 0: ./apm_brahma 38 38  1  25 2    (kernels 1-25, deviation 2)
#   GPU 1: ./apm_brahma 38 38  26 50 2    (kernels 26-50, deviation 2)
#   GPU 2: ./apm_brahma 38 38  51 75 2    (kernels 51-75, deviation 2)
#   GPU 3: ./apm_brahma 38 38  76 100 2   (kernels 76-100, deviation 2)
# ==============================================================================

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 GROUP TOTAL_KERNELS NUM_GPUS DEVIATION [WALLTIME]"
    echo ""
    echo "  GROUP          - Prime group number (e.g. 38)"
    echo "  TOTAL_KERNELS  - Total kernel files (e.g. 100)"
    echo "  NUM_GPUS       - Number of GPU jobs to submit (e.g. 4)"
    echo "  DEVIATION      - Deviation level to run (e.g. 2)"
    echo "  WALLTIME       - Optional walltime (default: 24:00:00)"
    echo ""
    echo "Example:"
    echo "  $0 38 100 4 2              # 4 GPUs, deviation 2"
    echo "  $0 38 100 4 3 48:00:00     # 4 GPUs, deviation 3, 48h walltime"
    exit 1
fi

GROUP=$1
TOTAL=$2
GPUS=$3
DEV=$4
WALLTIME=${5:-"24:00:00"}
CHUNK=$((TOTAL / GPUS))

# Create logs directory
mkdir -p logs

echo "============================================================"
echo " Parallel APM Submission"
echo "============================================================"
echo "  Group          : $GROUP"
echo "  Total kernels  : $TOTAL"
echo "  GPU jobs       : $GPUS"
echo "  Deviation      : $DEV (FIXED — same for all GPUs)"
echo "  Chunk size     : ~$CHUNK kernels per GPU"
echo "  Walltime       : $WALLTIME"
echo "============================================================"
echo ""

for i in $(seq 0 $((GPUS-1))); do
    KMIN=$((i * CHUNK + 1))
    KMAX=$(((i+1) * CHUNK))

    # Last chunk gets any remainder
    if [ $i -eq $((GPUS-1)) ]; then
        KMAX=$TOTAL
    fi

    JOB_NAME="apm_g${GROUP}_d${DEV}_k${KMIN}_${KMAX}"
    LOG_FILE="logs/g${GROUP}_d${DEV}_k${KMIN}_${KMAX}.txt"

    echo "  Submitting: $JOB_NAME  (kernels $KMIN to $KMAX, dev $DEV)"

    cat << EOF | qsub
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -q gpu
#PBS -l walltime=${WALLTIME}
#PBS -o logs/${JOB_NAME}.stdout
#PBS -e logs/${JOB_NAME}.stderr

cd \$PBS_O_WORKDIR

echo "=== Job: ${JOB_NAME} ==="
echo "Start: \$(date)"
echo "Node:  \$(hostname)"
echo "GPU:   \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Args:  group=${GROUP} kernels=${KMIN}-${KMAX} deviation=${DEV}"
echo ""

CUDA_VISIBLE_DEVICES=0 ./apm_brahma ${GROUP} ${GROUP} ${KMIN} ${KMAX} ${DEV} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "End: \$(date)"
EOF

done

echo ""
echo "============================================================"
echo " $GPUS jobs submitted for group $GROUP, deviation $DEV"
echo " Logs will be in: logs/"
echo "============================================================"
echo ""
echo "Monitor with:  qstat -u \$USER"
echo "Check results: python3 check_results.py $GROUP -d $DEV"
echo ""
echo "Workflow:"
echo "  1. Wait for all jobs to finish"
echo "  2. python3 check_results.py $GROUP -d $DEV"
echo "  3. If all hit → done. If not → run next deviation:"
echo "     ./submit_parallel.sh $GROUP $TOTAL $GPUS $((DEV+1))"
