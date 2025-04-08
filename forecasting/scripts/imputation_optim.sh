#! /bin/bash

# Only quantiles used for rectangular PR
timeout=45
predict_shift=112
prune_memory_error=true
M=false
r=true # reverse future (bidirectional LSTM)
for T in 4 # max_train_days
do for s in 1 #0 # 1 #2
do
    id=stdin/imputation_optim_s-"$s"_M-"$M"_T-"$T"
    run -c 1 -m 60 -g 1 -G L40S -t 48:0 -o "$id".out -e "$id".err "python imputation_optim.py -s $s -M $M -T $T -t $timeout -S $predict_shift -p $prune_memory_error -r $r"
done
done

# All quantiles
timeout=60
predict_shift=112
prune_memory_error=true
M=true
r=true # reverse future (bidirectional LSTM)
q=all # all quantiles
T=4 # max_train_days
for Q in 5 # 0.5 1 # s_q
do for s in 0 # 1 #2
do
    id=stdin/imputation_optim_s-"$s"_M-"$M"_T-"$T"_q-"$q"_Q-"$Q"
    run -c 1 -m 60 -g 1 -G L40S -t 63:0 -o "$id".out -e "$id".err "python imputation_optim.py -s $s -M $M -T $T -t $timeout -S $predict_shift -p $prune_memory_error -r $r -q $q -Q $Q"
done
done

# Target correlation
P=true
timeout=46
predict_shift=112
prune_memory_error=true
M=true
r=true # reverse future (bidirectional LSTM)
T=4 # max_train_days
s=0
id=stdin/imputation_optim_s-"$s"_M-"$M"_T-"$T"_P-"$P"
run -c 1 -m 48 -g 1 -t 48:0 -o "$id".out -e "$id".err "python imputation_optim.py -s $s -M $M -T $T -t $timeout -S $predict_shift -p $prune_memory_error -r $r -P $P"
