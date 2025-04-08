#! /bin/bash

# Only quantiles used for rectangular PR
w=all
patience=10
epochs=200
L=false # worse results when true
E=true # worse results when false
D=false # false
predict_shift=112
r=true # reverse future
M=true # false
for T in 4 #1 3 5 9 #7 14 21 # 56 84
do for i in 0 1 2 3 4
do for m in quantile
do
    id=stdin/imputation_w-"$w"_i-"$i"_m-"$m"_E-"$E"_L-"$L"_M-"$M"_D-"$D"_T-"$T"_predict_shift-"$predict_shift"_r-"$r"
    run -c 1 -g 1 -m 24 -t 20:00 -o "$id".out -e "$id".err "python imputation_store.py -w $w -i $i -m $m -E $E -L $L -M $M -D $D -P $patience -e $epochs -T $T -S $predict_shift -r $r -O true"
done
done
done

# All quantiles (TFT[B])
q=all # all quantiles
w=all
patience=40
epochs=200
L=false # worse results when true
E=true # worse results when false
D=false # false
predict_shift=112
r=true # reverse future
M=true # false
T=4 # max_train_days
for Q in 5 # 0.5 1 # s_q
do for i in 0 1 2 3 4
do for m in quantile
do
    id=stdin/imputation_w-"$w"_i-"$i"_m-"$m"_E-"$E"_L-"$L"_M-"$M"_D-"$D"_T-"$T"_predict_shift-"$predict_shift"_r-"$r"_Q-"$Q"_q-"$q"
    run -c 1 -g 1 -m 24 -t 20:00 -o "$id".out -e "$id".err "python imputation_store.py -w $w -i $i -m $m -E $E -L $L -M $M -D $D -P $patience -e $epochs -T $T -S $predict_shift -r $r -Q $Q -q $q"
done
done
done

# OTHER TRAINING DATASET SIZES
w=all
L=false # worse results when true
E=true # worse results when false
D=false # false
predict_shift=112
r=true # reverse future
M=true # false
T=4
i=best
m=quantile
for s in 0 1 2 3 4
do for R in 0.45 0.72 0.86 0.93  0.97 0.98 # n=200, n=100, n=50, n=25, n=10, n=5
do
    id=stdin/imputation_w-"$w"_i-"$i"_m-"$m"_E-"$E"_L-"$L"_M-"$M"_D-"$D"_T-"$T"_predict_shift-"$predict_shift"_r-"$r"_R-"$R"_s-"$s"
    run -c 1 -g 1 -m 24 -t 13:00 -o "$id".out -e "$id".err "python imputation_store.py -w $w -i $i -m $m -E $E -L $L -M $M -D $D -T $T -S $predict_shift -r $r -R $R -s $s"
done
done
