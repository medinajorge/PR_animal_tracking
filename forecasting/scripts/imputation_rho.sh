#! /bin/bash

# Target rho
V=true # rho
w=all
patience=40
epochs=200
L=false # worse results when true
E=true # worse results when false
D=false # false
predict_shift=112
r=true # reverse future
M=true # false
T=4 # max_train_days
m=quantile
H="quantiles=all,s_q=5" # mod kwargs for hyperparameters
for i in 0
do
    id=stdin/imputation_RHO_i-"$i"
    run -c 10 -m 22 -t 70:00 -o "$id".out -e "$id".err "python imputation_store.py -w $w -i $i -m $m -E $E -L $L -M $M -D $D -P $patience -e $epochs -T $T -S $predict_shift -r $r -H $H -V $V"
done
