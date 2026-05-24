#!/bin/bash
# 7 model variants x 4 datasets x 3 seeds = 84 runs (original block)
# + Big-Kitsune teacher sweep: 3 ratios x (1 teacher + 3 routings) x 4 datasets x 3 seeds = 144 extra runs
set -e

DATASETS="optdigits mnist landsat backdoor"
SEEDS="42 123 456"

# --- Big-Kitsune sweep config ---
# Full capacity sweep. r=4 is the middle ground; r=16 and r=32 push the teacher
# into the "wider than vanilla AE per-group hidden" regime (vanilla-AE-matched
# r* is dataset-dependent: ~30 for backdoor, ~190 for landsat).
BIGKIT_RATIOS="4.0 16.0 32.0"
BIGKIT_ROUTINGS="concat per_subae subae_paired"

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

        # B1/B2: Big-Kitsune teacher (sweep) + Kitsune + PKT (sweep over routings)
        for r in $BIGKIT_RATIOS; do
            python -m training.train_teacher --dataset $dataset --seed $seed \
                --model bigkit --hidden_ratio $r
            for routing in $BIGKIT_ROUTINGS; do
                python -m training.train_kitsune --dataset $dataset --seed $seed --mode pkt \
                    --teacher bigkit --teacher_hidden_ratio $r --pkt_mode $routing
            done
        done
    done
done

echo "=== All training complete. Running evaluation pass... ==="
python scripts/evaluate_all.py

echo "=== All runs complete. Results saved to results/ ==="
