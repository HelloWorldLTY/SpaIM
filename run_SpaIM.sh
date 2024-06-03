# /usr/bin/env bash

sp='checkpoint_SpaIM'
lr=1e-3
ep=300

for i in {1..45}
do
    ds="benchmark_datasets/Dataset$i"
    echo "Current dataset: $ds"
    val=0
    for kfold in {0..9}
    do
        python3 main_benchmark.py --kfold $kfold --dataset_name $ds --val_only $val --save_path $sp --epochs $ep
    done
    python3 eval_results.py --dataset_name $ds --save_path $sp
done


# Train nanostring dataset
val=0
lr=1e-3
ep=300
root='dataset/'

for ds in 'nano5-1' 'nano5-2' 'nano5-3' 'nano6'  'nano9-1' 'nano9-2' 'nano12' 'nano13'
do
    echo "Current dataset: $ds"
    ps="results/$sp"

    for kfold in {0..10}
    do
        python3 main.py --kfold $kfold --root $root --val_only $val --save_path $ps --lr $lr --epochs $ep --dataset_name $ds
    done
done
