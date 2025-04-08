#! /bin/bash

d=qrde # density
E=false # exclude median
S=true # use rho_spread for optimization of Q
K=true # optimize cmax
for R in true # false
do for t in forecasting
do for Q in 0.5 1 5 10
do for p in 0 1 2 3 4
do for m in contour # hull #alpha_shape
do
    if [ "$m" == "contour" ]; then
        time=4:30
    else
        time=0:40
    fi
    for P in val test
    do for M in true # false # mpl_val
    do
        id=stdin/dist_pr_"$t"_Q"$Q"_p"$p"_m-"$m"_P-"$P"_M-"$M"_R-"$R"_d-"$d"_E-"$E"_S-"$S"_K-"$K"
        run -c 1 -m 35 -t "$time" -e "$id".err -o "$id".out "python dist_pr.py -t $t -Q $Q -p $p -m $m -P $P -M $M -R $R -d $d -E $E -S $S -K $K"
    done
    done
done
done
done
done
done


d=qrde # density
E=false # exclude median
S=true # use rho_spread for optimization of Q
K=true # optimize cmax
for R in true #false
do for t in imputation
do for Q in 5 1 0.5
do for p in 1 3 4 0 2
do for m in contour hull
do
    if [ "$m" == "contour" ]; then
        time=4:35
    else
        time=0:45
    fi
    for P in val test
    do for M in true # false # mpl_val
    do
        id=stdin/dist_pr_"$t"_Q"$Q"_p"$p"_m-"$m"_P-"$P"_M-"$M"_R-"$R"_d-"$d"_E-"$E"_S-"$S"_K-"$K"
        run -c 1 -m 35 -t "$time" -e "$id".err -o "$id".out "python dist_pr.py -t $t -Q $Q -p $p -m $m -P $P -M $M -R $R -d $d -E $E -S $S -K $K"
    done
    done
done
done
done
done
done

# Prediction region rainbow
d=qrde # density
c=all # confidences
S=true # use rho_spread for optimization of Q
K=true # optimize cmax
for o in cds quality
do for R in true
do for t in forecasting imputation
do for Q in 5
do for p in 0
do for m in contour # hull #alpha_shape
do
    if [ "$m" == "contour" ]; then
        time=18:0
    else
        time=8:0
    fi
    for P in test # val test
    do for M in true # false # mpl_val
    do
        id=stdin/dist_pr_"$t"_Q"$Q"_p"$p"_m-"$m"_P-"$P"_M-"$M"_R-"$R"_d-"$d"_c-"$c"_o-"$o"_S-"$S"_K-"$K"
        run -c 1 -m 35 -t "$time" -e "$id".err -o "$id".out "python dist_pr.py -t $t -Q $Q -p $p -m $m -P $P -M $M -R $R -d $d -c $c -o $o -S $S -K $K"
    done
    done
done
done
done
done
done
done
