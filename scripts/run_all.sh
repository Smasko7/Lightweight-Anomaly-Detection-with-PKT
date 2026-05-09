#!/bin/bash
# 7 model variants x 3 datasets x 3 seeds = 63 runs
set -e

DATASETS="optdigits speech mnist landsat backdoor"
SEEDS="42 123 456"

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        echo "=== $dataset | seed=$seed ==="

        # K0: Kitsune baseline (no KD)
        python -m training.train_kitsune --dataset $dataset --seed $seed --mode vanilla

        # V1: Vanilla AE teacher
        python -m training.train_teacher --dataset $dataset --seed $seed --model vanilla

        # V2: Kitsune + Vanilla AE PKT
        python -m training.train_kitsune --dataset $dataset --seed $seed --mode pkt --teacher vanilla

        # C1: CNN teacher
        python -m training.train_teacher --dataset $dataset --seed $seed --model cnn

        # C2: Kitsune + CNN PKT
        python -m training.train_kitsune --dataset $dataset --seed $seed --mode pkt --teacher cnn

        # T1: Transformer teacher
        python -m training.train_teacher --dataset $dataset --seed $seed --model transformer

        # T2: Kitsune + Transformer PKT
        python -m training.train_kitsune --dataset $dataset --seed $seed --mode pkt --teacher transformer
    done
done

echo "=== All training complete. Running evaluation pass... ==="
python scripts/evaluate_all.py

echo "=== All runs complete. Results saved to results/ ==="
