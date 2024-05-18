#!/usr/bin/env bash

# Define datasets
datasets=("nano5-1" "nano5-2" "nano5-3" "nano6" "nanostring" "nano9-2" "nano12" "nano13")

# Run tangram
for dataset in "${datasets[@]}"; do
    python3 run_tangram.py --dataset "$dataset"
done

# Run gimvi
for dataset in "${datasets[@]}"; do
    python3 run_gimvi.py --dataset "$dataset"
done

# Run spage
for dataset in "${datasets[@]}"; do
    python3 run_spage.py --dataset "$dataset"
done
