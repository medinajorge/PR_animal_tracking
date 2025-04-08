#! /bin/bash

S=true # store missing idxs
for t in forecasting # imputation
do for p in val test
do for f in false #true
do for i in 0
do
    id=stdin/baseline_width-"$f"_i-"$i"_t-"$t"_p-"$p"_S-"$S"
    run -c 1 -m 2 -t 0:30 -o "$id".out -e "$id".err "python baseline_width.py -f $f -i $i -t $t -p $p -S $S"
done
done
done
done


# OTHER TRAINING LENGTHS
E=true
S=true
P=112
for T in 4
do for t in imputation #forecasting
do for p in val test
do for f in false #true
do for i in 0
do
    id=stdin/baseline_width-"$f"_i-"$i"_t-"$t"_p-"$p"_T-"$T"_E-"$E"_S-"$S"_P-"$P"
    run -c 1 -m 2 -t 0:30 -o "$id".out -e "$id".err "python baseline_width.py -f $f -i $i -t $t -p $p -T $T -E $E -S $S -P $P"
done
done
done
done
done
