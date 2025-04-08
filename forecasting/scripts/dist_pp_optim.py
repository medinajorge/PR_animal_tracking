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
task = 'forecasting'
params_idx = 'best'
s_q = 1
rho = False
density = 'pchip'

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:p:Q:P:d:",
                               ["task=", "params_idx=", "s_q=", "rho=", "density="])

except getopt.GetoptError:
    print('quality_across_seeds.py -n <n> -t <task>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-t", "--task"):
        task = arg
    elif opt in ("-p", "--params_idx"):
        params_idx = fmt.decoder(arg)
    elif opt in ("-Q", "--s_q"):
        s_q = fmt.decoder(arg)
    elif opt in ("-P", "--rho"):
        rho = fmt.decoder(arg)
    elif opt in ("-d", "--density"):
        density = arg


phdu.getopt_printer(opts)
kwargs = dict(task=task, params_idx=params_idx, s_q=s_q, rho=rho)
print(f"kwargs:\n{kwargs}")
input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
if task == 'forecasting':
    input_kwargs['max_train_days'] = 4
if density != 'pchip':
    input_kwargs['density'] = density
print(f"input_kwargs:\n{input_kwargs}")

df = analysis.optimize_mode_pp(**input_kwargs)
best = df.iloc[0]
print(f"best:\n{best}")
print("Computing test results")

# sample_stat, CI = analysis.point_prediction_aggregate_CI(**input_kwargs,
#                                                          quantiles='all', partition='test',
#                                                          dist_mode='mode', mode_margin='best')
# print(f"sample_stat:\n{sample_stat}")
# print(f"CI:\n{CI}")

print("Done!")
