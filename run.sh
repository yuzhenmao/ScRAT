#!/bin/bash

run_base()
{
    python main.py --task stage --train_sample_cells=200  --test_sample_cells=200  --train_num_sample=20 --test_num_sample=20  --model $1 --dir $2
    python main.py --task severity --train_sample_cells=200  --test_sample_cells=200  --train_num_sample=20 --test_num_sample=20  --model $1 --dir $2

    python main.py --task stage --train_sample_cells=500  --test_sample_cells=500  --train_num_sample=20 --test_num_sample=20  --model $1 --dir $2
    python main.py --task severity --train_sample_cells=500  --test_sample_cells=500  --train_num_sample=20 --test_num_sample=20  --model $1 --dir $2

    python main.py --task stage --train_sample_cells=200  --test_sample_cells=200  --train_num_sample=50 --test_num_sample=50  --model $1 --dir $2
    python main.py --task severity --train_sample_cells=200  --test_sample_cells=200  --train_num_sample=50 --test_num_sample=50  --model $1 --dir $2

    python main.py --task stage --train_sample_cells=500  --test_sample_cells=500  --train_num_sample=50 --test_num_sample=50  --model $1 --dir $2
    python main.py --task severity --train_sample_cells=500  --test_sample_cells=500  --train_num_sample=50 --test_num_sample=50  --model $1 --dir $2
}

#run_base Transformer Transformer3
run_base linear linear3


