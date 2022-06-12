#!/bin/bash

centers="5 10 15 20 25"
scripts/synthetic.py --variation $2 --split $3 --examples $4 --seed $5
python -m cloudpred data/signature/$2_$3_$4_$5/ -t log --logfile log/signature_$1_$2_$3_$4_$5 --cloudpred --linear --generative --genpat --deepset --centers ${centers} --dims $1 --valid 50 --test 50
rm -rf data/signature/$2_$3_$4_$5/ 
