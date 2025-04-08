#! /bin/bash

# Download ERA5 data and process it.

for year in $(seq 1985 1987)
do
    run -c 1 -m 10 -t 60:00 -o single_levels_"$year".out -e single_levels_"$year".err "python-jl era5_single-levels-hour.py -y $year"
done

step=8
for year in 1988 1997 2006 2015
    do year_arr=($year)
    for increment in $(seq 1 $(($step - 1)))
        do year_arr+=","$(($year + $increment))
    done
    run -c 1 -m 80 -t 520:30 -o single_levels_"$year"_full.out -e single_levels_"$year"_full.err "python-jl era5_single-levels-hour.py -y $year_arr"
done
