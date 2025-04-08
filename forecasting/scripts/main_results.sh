#! /bin/bash

for t in forecasting imputation
do for e in false true
do
    id=stdin/main_results_"$t"_CI_exp-"$e"
    run -c 1 -m 8 -t 1:00 -e "$id".err -o "$id".out "python main_results.py -t $t -e $e"
done
done
