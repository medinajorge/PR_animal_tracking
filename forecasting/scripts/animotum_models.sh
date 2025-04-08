#! /bin/bash

# Batch computation
for m in rw crw #mp # mp will raise error: Hessian was not positive-definite so some standard errors could not be calculated. Use single IDs
do for t in forecasting imputation
do for p in val test
do
    id=stdin/animotum_"$m"_"$t"_"$p"
    run -c 1 -m 13 -t 8:00 -o "$id".out -e "$id".err "Rscript --no-save animotum_models.R --model $m --task $t --test-partition $p"
done
done
done


# PF ONLY
for m in rw
do for t in imputation #forecasting
do for p in val test
do for days in 7
do for b in TRUE #FALSE
do
    id=stdin/animotum_"$m"_"$t"_"$p"_PF_ONLY_"$days"_baseline-"$b"
    run -c 1 -m 6 -t 1:00 -o "$id".out -e "$id".err "Rscript --no-save animotum_models.R --model $m --task $t --test-partition $p --pf-only TRUE --test-days $days --baseline $b"
done
done
done
done
done


# Individual IDS
for m in mp rw crw
do for t in forecasting imputation
do for p in val test
do
    id=stdin/animotum_individual_IDS_"$m"_"$t"_"$p"
    run -c 1 -m 13 -t 40:00 -o "$id".out -e "$id".err "Rscript --no-save animotum_models.R --model $m --task $t --individual-ids TRUE --test-partition $p"
done
done
done
