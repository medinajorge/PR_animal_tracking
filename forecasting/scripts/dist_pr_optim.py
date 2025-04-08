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
method = 'contour'
overwrite = False
c = 0.9
rho = False
density = 'pchip'
cds = 'mercator'
monotonic_q = False
exclude_me = False

optimize_spread = False
optimize_cmax = False

n_sample = None
n_grid = 50

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:p:Q:m:O:c:P:d:C:M:E:S:K:N:n:",
                               ["task=", "params_idx=", "s_q=", "method=", "overwrite=", "c=","rho=", "density=", "cds=", "monotonic_q=", "exclude_me=", "optimize_spread=", "optimize_cmax=", "n_sample=", "n_grid="])

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
    elif opt in ("-m", "--method"):
        method = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
    elif opt in ("-c", "--c"):
        c = fmt.decoder(arg)
    elif opt in ("-P", "--rho"):
        rho = fmt.decoder(arg)
    elif opt in ("-d", "--density"):
        density = arg
    elif opt in ("-C", "--cds"):
        cds = arg
    elif opt in ("-M", "--monotonic_q"):
        monotonic_q = fmt.decoder(arg)
    elif opt in ("-E", "--exclude_me"):
        exclude_me = fmt.decoder(arg)
    elif opt in ("-S", "--optimize_spread"):
        optimize_spread = fmt.decoder(arg)
    elif opt in ("-K", "--optimize_cmax"):
        optimize_cmax = fmt.decoder(arg)
    elif opt in ("-N", "--n_sample"):
        n_sample = fmt.decoder(arg)
    elif opt in ("-n", "--n_grid"):
        n_grid = fmt.decoder(arg)

phdu.getopt_printer(opts)

kwargs = dict(task=task, params_idx=params_idx, s_q=s_q, method=method, overwrite=overwrite, target_cs=[c], rho=rho, n_sample=n_sample, n_grid=n_grid)
print(f"kwargs:\n{kwargs}")

input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
if task == 'forecasting':
    input_kwargs['max_train_days'] = 4
if density != 'pchip':
    input_kwargs['density'] = density
if cds != 'mercator':
    input_kwargs['cds'] = cds
if monotonic_q:
    input_kwargs['monotonic_q'] = monotonic_q
if exclude_me:
    input_kwargs['exclude_me'] = exclude_me
if rho and optimize_spread:
    input_kwargs['optimize_spread'] = optimize_spread
if rho and optimize_cmax:
    input_kwargs['optimize_cmax'] = optimize_cmax
print(f"input_kwargs:\n{input_kwargs}")

result = analysis.dist_pr_mpl_val(**input_kwargs)

print(result)

print("Done!")
