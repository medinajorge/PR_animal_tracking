#! /bin/bash

# FORECASTING
for c in true false
do for T in forecasting # imputation
do for t in distance area
do for e in false #true false
do for a in true # false
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=30
        runtime=30:00
    else
        mem=6
        runtime=10:00
    fi
    id=stdin/error_analysis_shap_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"
    run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_shap.py -T $T -t $t -e $e -a $a -c $c"
done
done
done
done
done

# IMPUTATION
for c in false # true #true
do for T in imputation
do for t in distance area
do for e in true #true false
do for a in false #true # false
do for E in replace
do
    if [ "$T" == "imputation" ] && [ "$c" == "true" ]; then
        mem=30
        runtime=30:00
    else
        mem=6
        runtime=10:00
    fi
    id=stdin/error_analysis_shap_"$T"_"$t"_e-"$e"_a-"$a"_c-"$c"_E-"$E"
    run -c 1 -m $mem -t $runtime -e "$id".err -o "$id".out "python error_analysis_shap.py -T $T -t $t -e $e -a $a -c $c -O true -E $E"
done
done
done
done
done
done
