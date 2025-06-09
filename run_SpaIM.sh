# /usr/bin/env bash

sp='SpaIM_results'  # path to save
lr=1e-3  # key value to adjust
ep=300   # key value to adjust

for i in {1..53}
do
    ds="Dataset$i"
    echo "Current dataset: $ds"
    val=0
    for kfold in {0..9}
    do
        python3 src/main.py --kfold $kfold --dataset_name $ds --val_only $val --save_path $sp --epochs $ep
    done
    # python3 eval_results.py --dataset_name $ds --save_path $sp
done
