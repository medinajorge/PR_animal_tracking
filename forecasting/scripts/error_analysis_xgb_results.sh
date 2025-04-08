#! /bin/bash

# FORECASTING
# optimal features (all features false)
for c in true false
do for T in forecasting
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=20
        runtime=2:00
    else
        mem=4
        runtime=1:00
    fi
    for t in Q distance #Q_alpha area
    do for e in true false
    do for a in false # false
    do
        id=stdin/error_analysis_xgb_results_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"
        run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_xgb_results.py -T $T -t $t -e $e -a $a -c $c -O true"
    done
    done
    done
done
done

# all features
for c in true false
do for T in forecasting
do for t in Q distance Q_alpha area
do for e in true false #true false
do for a in true # false
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=20
        runtime=2:00
    else
        mem=4
        runtime=1:00
    fi
    id=stdin/error_analysis_xgb_results_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"
    run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_xgb_results.py -T $T -t $t -e $e -a $a -c $c -O true"
done
done
done
done
done


################################################
# IMPUTATION
# optimal features (all features false)
for c in true false # use correlations or not
do for T in imputation
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=20
        runtime=2:00
    else
        mem=4
        runtime=1:00
    fi
    for t in area distance Q #distance #Q_alpha area
    do for e in true false #true #false
    do for a in false # false
    do for E in none add replace # default is None
    do
        id=stdin/error_analysis_xgb_results_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"_E-"$E"
        run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_xgb_results.py -T $T -t $t -e $e -a $a -c $c -E $E -O true"
    done
    done
    done
    done
done
done

# all features
for c in true false
do for T in imputation
do for E in none add replace
do for t in Q distance Q_alpha area
do for e in true false #true false
do for a in true # false
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=20
        runtime=2:00
    else
        mem=4
        runtime=1:00
    fi
    id=stdin/error_analysis_xgb_results_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"_E-"$E"
    run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_xgb_results.py -T $T -t $t -e $e -a $a -c $c -O true -E $E"
done
done
done
done
done
done
