#! /bin/bash

for T in forecasting
do for c in true false
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]
    then
        runtime=90
    else
        runtime=6
    fi
    for e in true false
    do for t in area Q distance
    do
        id=stdin/error_analysis_mrmr_"$T"_"$t"_e-"$e"_c-"$c"
        run -c 1 -m 8 -t "$runtime":00 -e "$id".err -o "$id".out "python error_analysis_feature_selection.py -T $T -t $t -e $e -c $c"
    done
    done
done
done

for T in imputation
do for E in none add replace
do for c in true false
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]
    then
        runtime=90
    else
        runtime=2
    fi
    for e in true false
    do for t in area Q distance
    do
        id=stdin/error_analysis_mrmr_"$T"_"$t"_e-"$e"_c-"$c"_E-"$E"
        run -c 1 -m 8 -t "$runtime":00 -e "$id".err -o "$id".out "python error_analysis_feature_selection.py -T $T -t $t -e $e -c $c -E $E"
    done
    done
done
done
done
