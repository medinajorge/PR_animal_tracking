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
partition = 'test'
mpl_val = False
density = 'pchip'

rho = False

cds = 'mercator'
monotonic_q = False
exclude_me = False

cs = 'default'
output = 'quality'

optimize_spread = False
optimize_cmax = False

dive_data = False
add_z = True

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:p:Q:m:O:P:M:R:d:C:X:E:c:o:S:K:D:z:",
                               ["task=", "params_idx=", "s_q=", "method=", "overwrite=", "partition=", "mpl_val=", "rho=", "density=", "cds=", "monotonic_q=", "exclude_me=", "cs=", "output=", "optimize_spread=", "optimize_cmax=", "dive_data=", "add_z="])

except getopt.GetoptError:
    print('dist_pr.py -t <task> -p <params_idx> -Q <s_q> -m <method> -O <overwrite> -P <partition> -M <mpl_val>')
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
    elif opt in ("-P", "--partition"):
        partition = arg
    elif opt in ("-M", "--mpl_val"):
        mpl_val = fmt.decoder(arg)
    elif opt in ("-R", "--rho"):
        rho = fmt.decoder(arg)
    elif opt in ("-d", "--density"):
        density = arg
    elif opt in ("-C", "--cds"):
        cds = arg
    elif opt in ("-X", "--monotonic_q"):
        monotonic_q = fmt.decoder(arg)
    elif opt in ("-E", "--exclude_me"):
        exclude_me = fmt.decoder(arg)
    elif opt in ("-c", "--cs"):
        cs = arg
    elif opt in ("-o", "--output"):
        output = arg
    elif opt in ("-S", "--optimize_spread"):
        optimize_spread = fmt.decoder(arg)
    elif opt in ("-K", "--optimize_cmax"):
        optimize_cmax = fmt.decoder(arg)
    elif opt in ("-D", "--dive_data"):
        dive_data = fmt.decoder(arg)
    elif opt in ("-z", "--add_z"):
        add_z = fmt.decoder(arg)

phdu.getopt_printer(opts)
kwargs = dict(task=task, params_idx=params_idx, s_q=s_q, method=method, overwrite=overwrite, partition=partition, mpl_val=mpl_val)
print(f"kwargs:\n{kwargs}")

input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
if task == 'forecasting':
    input_kwargs['max_train_days'] = 4
if method == 'alpha_shape':
    input_kwargs['n_sample'] = int(1e4)
if rho:
    input_kwargs['rho'] = rho
if density != 'pchip':
    input_kwargs['density'] = density
if cds != 'mercator':
    input_kwargs['cds'] = cds
if monotonic_q:
    input_kwargs['monotonic_q'] = monotonic_q
if exclude_me:
    input_kwargs['exclude_me'] = exclude_me
if cs == 'all':
    input_kwargs['cs'] = cs
if rho and optimize_spread:
    input_kwargs['optimize_spread'] = optimize_spread
if rho and optimize_cmax:
    input_kwargs['optimize_cmax'] = optimize_cmax

if dive_data:
    input_kwargs['n_sample'] = int(1e4)
    input_kwargs['preload'] = False
    del input_kwargs['max_train_days']
    if task == 'forecasting':
        dive_kws = {'weather': 'all', 'epochs': 200, 'cds': 'mercator', 'overwrite': False, 'patience': 20, 'mod_hp': {}, 'store_missing_idxs': True, 's_q': 3, 'dive_data': True, 'limit_train_batches': 400}
    elif task == 'imputation':
        dive_kws = {'weather': 'all', 'epochs': 200, 'cds': 'mercator', 'overwrite': False, 'task': 'imputation', 'mod_hp': {}, 'patience': 20, 'expand_encoder_until_future_length': True, 'store_missing_idxs': True, 'predict_shift': 128, 'reverse_future': True, 's_q': 3, 'dive_data': True, 'limit_train_batches': 400}
    else:
        raise ValueError(f"task={task} not recognized")
    input_kwargs.update(dive_kws)
    input_kwargs['add_z'] = add_z

print(f"input_kwargs:\n{input_kwargs}")
if output == 'quality':
    func = analysis.dist_pr
elif output == 'cds':
    func = lambda *args, **kwargs: analysis.dist_pr_cds(*args, **kwargs, save=True)
else:
    raise ValueError(f"output={output} not recognized")

print(f"Computing {output}...")
df = func(**input_kwargs)

if output == 'quality':
    result = df.groupby('confidence').mean().round(2)
    print(result)

print("Done!")
