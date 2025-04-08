#! /bin/bash

# Only quantiles used for rectangular PR
M=true # store_missing_idxs
O=true # overwrite
for i in 0 1 2 3 4
do
    id=stdin/forecast_i-"$i"_M-"$M"
    run -c 1 -g 1 -m 24 -t 13:00 -o "$id".out -e "$id".err "python forecast_store.py -i $i -M $M -O $O"
done

# All quantiles (TFT[B])
M=true # store_missing_idxs
q=all
T=4 # max_train_days
for Q in 0.5 #1 10 5
do for i in 0 1 2 3 4
do
    id=stdin/forecast_i-"$i"_M-"$M"_q-"$q"_Q-"$Q"_T-"$T"
    run -c 1 -g 1 -m 24 -t 13:00 -o "$id".out -e "$id".err "python forecast_store.py -i $i -M $M -q $q -Q $Q -T $T"
done
done

# Variable training dataset sizes
M=true # store_missing_idxs
O=false # overwrite
i=best
for S in 0 1 2 3 4
do for D in 0.45 0.72 0.86 0.93  0.97 0.98 # n=200, n=100, n=50, n=25, n=10, n=5
do
    id=stdin/forecast_i-"$i"_M-"$M"_O-"$O"_S-"$S"_D-"$D"
    run -c 1 -g 1 -m 24 -t 3:00 -o "$id".out -e "$id".err "python forecast_store.py -i $i -M $M -O $O -S $S -D $D"
done
done
