#! /bin/bash

# Only quantiles used for rectangular PR
timeout=60
prune_memory_error=true
M=true
for cds in mercator
do for w in all
do for s in 0
do
    id=stdin/forecast_optim_w-"$w"_s-"$s"_cds-"$cds"_timeout-"$timeout"_prune_memory_error-"$prune_memory_error"_M-"$M"
    run -c 1 -m 64 -g 1 -G L40S -t 63:0 -o "$id".out -e "$id".err "python forecast_optim.py -w $w -s $s -c $cds -t $timeout -P $prune_memory_error -M $M"
done
done
done

# All quantiles
timeout=46
prune_memory_error=true
M=true
q=all
D=4 # max_train_days
for cds in mercator
do for w in all
do for s in 0 #0 1 2
do for Q in 0.5 #1 5 10
do
    id=stdin/forecast_optim_w-"$w"_s-"$s"_cds-"$cds"_timeout-"$timeout"_prune_memory_error-"$prune_memory_error"_M-"$M"_Q-"$Q"_D-"$D"
    run -c 1 -m 48 -g 1 -t 48:0 -o "$id".out -e "$id".err "python forecast_optim.py -w $w -s $s -c $cds -t $timeout -P $prune_memory_error -M $M -q $q -Q $Q -D $D"
done
done
done
done

# Target rho (corr(X,Y))
p=true
timeout=46
prune_memory_error=true
M=true
D=4 # max_train_days
cds=mercator
w=all
s=0
id=stdin/forecast_optim_w-"$w"_s-"$s"_cds-"$cds"_timeout-"$timeout"_prune_memory_error-"$prune_memory_error"_M-"$M"_p-"$p"_D-"$D"
run -c 1 -m 48 -g 1 -t 48:0 -o "$id".out -e "$id".err "python forecast_optim.py -w $w -s $s -c $cds -t $timeout -P $prune_memory_error -M $M -p $p -D $D"
