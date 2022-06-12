#!/bin/bash

# Run all simulation experiments

dim=25

for seed in `seq 25`;
do
    variation=1.0
    split=0.5
    for examples in 110 120 130 140 150 200 250 300 400 500 750 1000
    do
        scripts/run_signature.sh ${dim} ${variation} ${split} ${examples} ${seed}
    done

    examples=200
    for variation in `seq 0 0.1 0.9`;
    do
        split=0.5
        scripts/run_signature.sh ${dim} ${variation} ${split} ${examples} ${seed}
    done

    variation=1.0
    for split in `seq 0 0.1 1`;
    do
        scripts/run_signature.sh ${dim} ${variation} ${split} ${examples} ${seed}
    done

    variation=1.0
    for examples in 110 120 130 140 150 200 250 300 400 500 750 1000
    do
        scripts/run_interaction.sh ${dim} ${variation} ${examples} ${seed}
    done

    examples=200
    for variation in `seq 0 0.1 1`;
    do
        scripts/run_interaction.sh ${dim} ${variation} ${examples} ${seed}
    done
done
