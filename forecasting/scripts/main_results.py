import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import warnings
sys.stdout.flush()
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        from tqdm import tqdm # Probably runing on standard python terminal. If does not work => should be replaced by tqdm(x) = identity(x)

RootDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(RootDir)
from forecasting import analysis
from tidypath import fmt
import phdu
import getopt

# defaults
task = 'forecasting'
overwrite=False
CI_expansion = True

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:O:e:",
                               ["task=", "overwrite=", "CI_expansion="])
except getopt.GetoptError:
    print('main_results.py -t <task> -O <overwrite> -e <CI_expansion>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-t", "--task"):
        task = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-e", "--CI_expansion"):
        CI_expansion = fmt.decoder(arg)

phdu.getopt_printer(opts)
kwargs = dict(task=task, mpl_val=CI_expansion)
             #, overwrite=overwrite)
print(f"kwargs:\n{kwargs}")

df = analysis.main_results(**kwargs)

print("Proceeding to supplementary results")
kwargs_sup = dict(task=task, CI_expansion=CI_expansion)
print(f"kwargs_sup:\n{kwargs_sup}")

df = analysis.aggregate_summary(unit=False, divide_area=1e5, simplify=False, area_exp_fmt=False, **kwargs_sup)

print("Done!")
