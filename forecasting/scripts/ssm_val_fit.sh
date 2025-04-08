#! /bin/bash

for t in forecasting imputation
do for r in false # true
do for m in mp rw crw
do for o in true false
do for s in true false
do for n in 500
do
    id=stdin/ssm_val_fit_m-"$m"_o-"$o"_s-"$s"_n-"$n"_r-"$r"_t-"$t"
    run -c 1 -m 4 -t 2:00 -e "$id".err -o "$id".out "python ssm_val_fit.py -m $m -o $o -s $s -n $n -r $r -t $t"
done
done
done
done
done
done
