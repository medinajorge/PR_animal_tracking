#! /bin/bash

# PP Predictions
# o=pp # output
# for t in forecasting imputation
# do for R in true  # rho
# do
#     id=stdin/dist_best_models_t-"$t"_R-"$R"_o-"$o"
#     run -c 1 -m 80 -t 4:00 -e "$id".err -o "$id".out "python dist_best_models.py -t $t -R $R -o $o"
# done
# done

E=false # exclude median
S=true # optimize spread for rho
K=true # optimize cmax for rho
for t in forecasting imputation
do for M in true # false # mpl val
do for R in true # false # rho
do for density in qrde # pchip
do
    id=stdin/dist_best_models_t-"$t"_M-"$M"_R-"$R"_density-"$density"_E-"$E"_S-"$S"_K-"$K"
    run -c 1 -m 2 -t 1:00 -e "$id".err -o "$id".out "python dist_best_models.py -t $t -M $M -R $R -d $density -E $E -S $S -K $K"
done
done
done
done
