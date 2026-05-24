#!/bin/bash
# Vanilla-only PoC: K0 + V1 + V2 across 3 datasets x 3 seeds = 27 runs
# Run AFTER run_alpha_sweep.sh to determine ALPHA below.
#
# Set ALPHA to the value from the sweep where V2 > K0 without exceeding V1:
ALPHA=0.1    # from alpha sweep: best trade-off on optdigits/seed42

DATASETS="optdigits mnist landsat backdoor"
SEEDS="42 123 456"

set -e

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        echo "=== $dataset | seed=$seed ==="

        # K0: Kitsune baseline (no KD)
        python -m training.train_kitsune \
            --dataset $dataset --seed $seed --mode vanilla

        # V1: Vanilla AE teacher
        python -m training.train_teacher \
            --dataset $dataset --seed $seed --model vanilla

        # V2: Kitsune + Vanilla PKT (paper-faithful, best alpha from sweep)
        python -m training.train_kitsune \
            --dataset $dataset --seed $seed \
            --mode pkt --teacher vanilla \
            --alpha $ALPHA
    done
done

echo ""
echo "=== All 27 runs complete. Running evaluation pass... ==="
python scripts/evaluate_all.py

echo "=== Results saved to results/ ==="
