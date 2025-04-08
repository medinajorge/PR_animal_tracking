#! /bin/bash

# Target rho. Computation performed using CPU
M=true # store_missing_idxs
T=4 # max_train_days
V=true # rho
H="quantiles=all,s_q=10" # mod kwargs for hyperparameters
for i in 0 # 1 2 3 4
do
    if [ $i -eq 3 ]; then
        cpu=15
    else
        cpu=10
    fi
    id=stdin/forecast_RHO_i-"$i"_M-"$M"_T-"$T"
    run -c $cpu -m 16 -t 40:00 -o "$id".out -e "$id".err "python forecast_store.py -i $i -M $M -T $T -V $V -H $H"
done
