#! /bin/bash

NPROCS=4
NNETS=10

for arch in "share" "noshare"
do
    for points in "random" "dg0"
    do
        for penalty in 0 1E-1 1E-2 1E-4 1E-6
        do
            for m in 2
            do
                mpirun -np $NPROCS python train_tf_fm_P0pts.py -m $m -architecture $arch -points $points -penalty $penalty -nnets $NNETS
            done
        done
    done
done
