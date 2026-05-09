#!/bin/bash
# Alpha sweep: optdigits, seed=42, vanilla teacher
# Purpose: find effective alpha range for paper-faithful PKT (no dynamic normalization)
# Results saved to results/sweep/
set -e

DATASET="optdigits"
SEED="42"
ALPHAS="0.01 0.1 1.0 10.0 100.0"

echo "=== K0: Kitsune vanilla baseline ==="
python -m training.train_kitsune --dataset $DATASET --seed $SEED --mode vanilla

echo "=== V1: Vanilla AE teacher ==="
python -m training.train_teacher --dataset $DATASET --seed $SEED --model vanilla

for alpha in $ALPHAS; do
    echo "=== V2: Kitsune + Vanilla PKT | alpha=$alpha ==="
    python -m training.train_kitsune \
        --dataset $DATASET --seed $SEED \
        --mode pkt --teacher vanilla \
        --alpha $alpha \
        --tag "alpha${alpha}" \
        --out_dir results/sweep
done

echo ""
echo "=== Sweep complete. Results in results/sweep/ ==="
echo "K0 baseline: results/${DATASET}_kitsune_vanilla_seed${SEED}.json"
echo "V1 teacher:  results/${DATASET}_vanilla_seed${SEED}.json"
for alpha in $ALPHAS; do
    echo "V2 alpha=$alpha: results/sweep/${DATASET}_kitsune_pkt_vanilla_seed${SEED}_alpha${alpha}.json"
done
