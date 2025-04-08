#! /bin/bash

# FORECASTING
# C = false (E=none always)
n=1000
memory=40 # prior it was 12
for c in false # true false
do
    if [ "$c" = true ]; then
        runtime=65:00
        timeout=62
    else
        runtime=32:00
        timeout=30
    fi
    for T in forecasting
    do for t in area Q distance #Q_alpha area
    do for e in true false
    do
        id=stdin/error_analysis_hp_"$t"_"$n"_"$T"_e-"$e"_c-"$c"
        run -c 1 -m $memory -t $runtime -e "$id".err -o "$id".out "python error_analysis_hp.py -t $t -n $n -l $timeout -T $T -e $e -c $c"
    done
    done
    done
done

# IMPUTATION
n=1000
memory=40 # prior it was 12
for c in true # true false
do for E in none add replace
do
    if [ "$c" = true ]; then
        runtime=65:00
        timeout=62
    else
        runtime=32:00
        timeout=30
    fi
    for T in imputation
    do for t in area Q distance #Q_alpha
    do for e in true false
    do
        id=stdin/error_analysis_hp_"$t"_"$n"_"$T"_e-"$e"_c-"$c"_E-"$E"
        run -c 1 -m $memory -t $runtime -e "$id".err -o "$id".out "python error_analysis_hp.py -t $t -n $n -l $timeout -T $T -e $e -c $c -E $E"
    done
    done
    done
done
done
