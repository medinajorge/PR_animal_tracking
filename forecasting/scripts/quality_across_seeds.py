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
from forecasting import analysis, params
from tidypath import fmt
import phdu
import getopt

# defaults
n = 200
task = 'forecasting'
params_idx = 'best'

try:
    opts, args = getopt.getopt(sys.argv[1:], "n:t:p:",
                               ["n=", "task=", "params_idx="])

except getopt.GetoptError:
    print('quality_across_seeds.py -n <n> -t <task>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-n", "--n"):
        n = int(arg)
    elif opt in ("-t", "--task"):
        task = arg
    elif opt in ("-p", "--params_idx"):
        params_idx = fmt.decoder(arg)

phdu.getopt_printer(opts)
kwargs = dict(n=n, task=task, params_idx=params_idx)
print(f"kwargs:\n{kwargs}")
input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
print(f"input_kwargs:\n{input_kwargs}")

df = analysis.quality_aggregate_CI_delete_seeds(**input_kwargs)

print("Done!")
