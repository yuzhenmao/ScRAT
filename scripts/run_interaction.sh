#!/bin/bash

centers="5 10 15 20"
bin/synthetic.py --variation $2 --examples $3 --seed $4 --interaction
python -m cloudpred data/interaction/$2_$3_$4/ -t log --logfile log/interaction_$1_$2_$3_$4 --cloudpred --linear --generative --genpat --deepset --centers ${centers} --dims $1 --valid 50 --test 50
rm -rf data/interaction/$2_$3_$4/
