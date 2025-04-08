#! /bin/bash

# mem for P false is 35, else 8
t=forecasting
E=false # exclude median
S=true # use rho_spread for optimization of Q
K=true # optimize c_max
for d in qrde #pchip
do for m in contour #hull contour
do if [ "$m" = hull ]; then
        # time=7:30
        time=11:30
    else
        # time=13:30
        # time=15:30
        time=20:30
    fi
    for P in true # false
    do if [ "$P" = true ]; then
            if [ "$m" = hull ]; then
                mem=8
            else
                mem=12
            fi
        else mem=35
        fi
        for Q in 0.5 10 1 5
        do for p in 0 1 2 3 4
        do for c in 0.5 0.9 0.95
        do
            id=stdin/dist_pr_optim_"$t"_Q"$Q"_p"$p"_m-"$m"_c-"$c"_P-"$P"_d-"$d"_E-"$E"_S-"$S"_K-"$K"
            run -c 1 -m $mem -t $time -e "$id".err -o "$id".out "python dist_pr_optim.py -t $t -Q $Q -p $p -m $m -c $c -P $P -d $d -E $E -S $S -K $K"
        done
        done
        done
        done
    done
done


# Imputation
t=imputation
E=false # exclude median
S=true # use rho_spread for optimization of Q
K=true # optimize c_max
for d in qrde #pchip
do for m in hull # contour
do if [ "$m" = hull ]; then
        # time=7:40
        time=11:40
    else
        # time=13:40
        # time=15:40
        time=20:40
    fi
    for P in true # false
    do if [ "$P" = true ]; then
            if [ "$m" = hull ]; then
                mem=8
            else
                mem=12
            fi
        else mem=35
        fi
        for Q in 5 1 0.5
        do for p in 0 1 2 3 4
        do for c in 0.5 0.9 0.95
        do
            id=stdin/dist_pr_optim_"$t"_Q"$Q"_p"$p"_m-"$m"_c-"$c"_P-"$P"_d-"$d"_E-"$E"_S-"$S"_K-"$K"
            run -c 1 -m $mem -t $time -e "$id".err -o "$id".out "python dist_pr_optim.py -t $t -Q $Q -p $p -m $m -c $c -P $P -d $d -E $E -S $S -K $K"
        done
        done
        done
        done
    done
done

# Prediction region rainbow (PR with alpha=0.05, ..., 0.95) for plotting
S=true # use rho_spread for optimization of Q
K=true # optimize c_max
for t in forecasting imputation
do for d in qrde #pchip
do for m in contour # hull contour
do if [ "$m" = hull ]; then
        # time=7:30
        time=11:30
    else
        # time=9:30
        time=14:00
    fi
    for P in true # false
    do if [ "$P" = true ]; then
            mem=8
        else mem=35
        fi
        for Q in 5
        do for p in 0
        do for c in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.55 0.6 0.65 0.7 0.75 0.8 0.85
        do
            id=stdin/dist_pr_optim_"$t"_Q"$Q"_p"$p"_m-"$m"_c-"$c"_P-"$P"_d-"$d"_S-"$S"_K-"$K"
            run -c 1 -m $mem -t $time -e "$id".err -o "$id".out "python dist_pr_optim.py -t $t -Q $Q -p $p -m $m -c $c -P $P -d $d -S $S -K $K"
        done
        done
        done
        done
    done
done
done
