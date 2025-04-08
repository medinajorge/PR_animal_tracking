#! /bin/bash

id=stdin/target_scales
run -c 1 -m 8 -t 1:10 -o "$id".out -e "$id".err "python target_scales.py"
