#! /bin/bash

# Training on single trajectories
M=true
for ID in ct134-277-14 ct134-295-14 ct135-084BAT-14 ct140-154BAT-15 ct140-158BAT-15 ct140-159BAT-15 ct140-161-15 ct140-162-15 ct140-163-15 ct140-168-15 ct140-169-15 ct140-170-15 ct140-171-15 ct140-173-15 ct140-174-15 ct140-175-15 ct140-176-15 ct140-177-15 ct140-178-15 ct140-179-15 ct140-188-15 ct140-189-15 ct140-635BAT-13 ct140-690BAT-13 ct140-695BAT-13 ct140-699BAT-13 ct140-700BAT-13 ct140-956BAT-14 ft22-686-18 ft22-873-18 ft22-874-18 ft22-876-18 ft22-878-18 ft22-879-18 ft22-881-18
do for p in false #true
do for w in all
do for m in quantile
do
    id=stdin/forecast_single_ID-"$ID"_w-"$w"_m-"$m"_p-"$p"_M-"$M"
    run -c 1 -m 10 -t 30:00 -o "$id".out -e "$id".err "python forecast_store.py -I $ID -w $w -m $m -p $p -M $M"
done
done
done
done
