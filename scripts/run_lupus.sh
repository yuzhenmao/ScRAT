#!/bin/bash

centers="5 10 15 20 25"
for seed in `seq 25`
do
    python3 -m cloudpred data/lupus_pop -t log --logfile log/lupus_pop/cloudpred_${seed}  --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_
    python3 -m cloudpred data/lupus_pop -t log --logfile log/lupus_pop/linear_${seed}     --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_
    python3 -m cloudpred data/lupus_pop -t log --logfile log/lupus_pop/generative_${seed} --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_
    python3 -m cloudpred data/lupus_pop -t log --logfile log/lupus_pop/genpat_${seed}     --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_
    python3 -m cloudpred data/lupus_pop -t log --logfile log/lupus_pop/deepset_${seed}    --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_

    python3 -m cloudpred data/lupus      -t log --logfile log/lupus/cloudpred_${seed}      --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_
    python3 -m cloudpred data/lupus      -t log --logfile log/lupus/linear_${seed}         --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_
    python3 -m cloudpred data/lupus      -t log --logfile log/lupus/generative_${seed}     --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_
    python3 -m cloudpred data/lupus      -t log --logfile log/lupus/genpat_${seed}         --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_
    python3 -m cloudpred data/lupus      -t log --logfile log/lupus/deepset_${seed}        --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_

    python3 -m cloudpred data/mono        -t log --logfile log/mono/cloudpred_${seed}       --cloudpred --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_ --regression
    python3 -m cloudpred data/mono        -t log --logfile log/mono/linear_${seed}          --linear    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_ --regression
    python3 -m cloudpred data/mono        -t log --logfile log/mono/deepset_${seed}         --deepset   --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_ --regression
done

for seed in `seq 10 25`
do
    for train in 10 20 30 40 50 60 70
    do
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/cloudpred_${seed}_train_${train}  --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/linear_${seed}_train_${train}     --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/generative_${seed}_train_${train} --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/genpat_${seed}_train_${train}     --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/deepset_${seed}_train_${train}    --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_train_${train}_ --train_patients ${train}

        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/cloudpred_${seed}_train_${train}  --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/linear_${seed}_train_${train}     --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/generative_${seed}_train_${train} --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/genpat_${seed}_train_${train}     --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_train_${train}_ --train_patients ${train}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/deepset_${seed}_train_${train}    --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_train_${train}_ --train_patients ${train}

        python3 -m cloudpred data/mono    -t log --logfile log/mono/cloudpred_${seed}_train_${train} --cloudpred --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_train_${train} --regression --train_patients ${train}
        python3 -m cloudpred data/mono    -t log --logfile log/mono/linear_${seed}_train_${train}    --linear    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_train_${train} --regression --train_patients ${train}
        python3 -m cloudpred data/mono    -t log --logfile log/mono/deepset_${seed}_train_${train}   --deepset   --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_train_${train} --regression --train_patients ${train}
    done

    for cells in 1 10 50 100 250 500 1000 2000 3000 4000 5000 6000 7000 8000
    do
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/cloudpred_${seed}_cells_${cells}  --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/linear_${seed}_cells_${cells}     --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/generative_${seed}_cells_${cells} --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/genpat_${seed}_cells_${cells}     --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus_pop     -t log --logfile log/lupus_pop/deepset_${seed}_cells_${cells}    --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_pop_${seed}_cells_${cells}_ --cells ${cells}

        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/cloudpred_${seed}_cells_${cells}  --cloudpred  --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/linear_${seed}_cells_${cells}     --linear     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/generative_${seed}_cells_${cells} --generative --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/genpat_${seed}_cells_${cells}     --genpat     --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_cells_${cells}_ --cells ${cells}
        python3 -m cloudpred data/lupus     -t log --logfile log/lupus/deepset_${seed}_cells_${cells}    --deepset    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/lupus_${seed}_cells_${cells}_ --cells ${cells}

        python3 -m cloudpred data/mono    -t log --logfile log/mono/cloudpred_${seed}_cells_${cells} --cloudpred --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_cells_${cells} --regression --cells ${cells}
        python3 -m cloudpred data/mono    -t log --logfile log/mono/linear_${seed}_cells_${cells}    --linear    --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_cells_${cells} --regression --cells ${cells}
        python3 -m cloudpred data/mono    -t log --logfile log/mono/deepset_${seed}_cells_${cells}   --deepset   --centers ${centers} --dim 100 --seed ${seed} --valid 0.25 --test 0.25 --figroot fig/mono_${seed}_cells_${cells} --regression --cells ${cells}
    done
done

for method in Linear Generative Genpat Cloud DeepSet
do
    echo ${method}
    grep INFO log/lupus_pop_* | grep AUC | grep ${method} | awk '{ total += $6; count++ } END { print total/count }'
done
