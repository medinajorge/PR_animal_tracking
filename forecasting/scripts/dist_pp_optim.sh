#! /bin/bash

# for t in forecasting
# do for Q in 1 5
# do for p in 0 1 2 3 4
# do
#     id=stdin/dist_pp_optim_"$t"_Q"$Q"_p"$p"
#     run -c 1 -m 100 -t 20:00 -e "$id".err -o "$id".out "python dist_pp_optim.py -t $t -Q $Q -p $p"
# done
# done
# done

# Best
d=qrde # density
t=forecasting
Q=5
p=0
P=true # rho
id=stdin/dist_pp_optim_"$t"_Q"$Q"_p"$p"_P"$P"_d"$d"
run -c 1 -m 35 -t 10:30 -e "$id".err -o "$id".out "python dist_pp_optim.py -t $t -Q $Q -p $p -P $P -d $d"

# Best
d=qrde # density
t=imputation
Q=1
p=1
P=true # rho
id=stdin/dist_pp_optim_"$t"_Q"$Q"_p"$p"_P"$P"_d"$d"
run -c 1 -m 35 -t 10:30 -e "$id".err -o "$id".out "python dist_pp_optim.py -t $t -Q $Q -p $p -P $P -d $d"
