#! /bin/bash

id=stdin/bathymetry_speed_train
run -c 1 -m 20 -t 3:00 -e "$id".err -o "$id".out "python bathymetry_speed_train.py"
