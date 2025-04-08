#! /bin/bash

p=4
for t in forecasting
do for n in 200 100 50 25 10 5
do
    id=stdin/"$t"_n-"$n"
    run -c 1 -m 16 -t 1:00 -e "$id".err -o "$id".out "python quality_across_seeds.py -t $t -n $n -p $p"
done
done

p=1
for t in imputation
do for n in 200 100 50 25 10 5
do
    id=stdin/"$t"_n-"$n"
    run -c 1 -m 16 -t 1:00 -e "$id".err -o "$id".out "python quality_across_seeds.py -t $t -n $n -p $p"
done
done
