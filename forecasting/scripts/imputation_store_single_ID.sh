#! /bin/bash

# Training on single trajectories
w=all
L=false # worse results when true
E=true # worse results when false
D=false
predict_shift=112
M=true
m=quantile
p=false
r=true
for T in 4
do for ID in ct140-154BAT-15 ct140-158BAT-15 ct140-159BAT-15 ct140-162-15 ct140-163-15 ct140-166-15 ct140-168-15 ct140-169-15 ct140-170-15 ct140-171-15 ct140-172-15 ct140-173-15 ct140-174-15 ct140-178-15 ct140-179-15 ct140-188-15 ct140-189-15 ct140-635BAT-13 ct140-699BAT-13 ct140-700BAT-13 ct140-956BAT-14
do
    id=stdin/imputation_single_ID-"$ID"_w-"$w"_m-"$m"_T-"$T"_L-"$L"_E-"$E"_D-"$D"_M-"$M"_p-"$p"_r-"$r"
    run -c 1 -m 20 -t 80:00 -o "$id".out -e "$id".err "python imputation_store.py -I $ID -w $w -m $m -T $T -p $p -L $L -E $E -D $D -M $M -S $predict_shift -r $r"
done
done
