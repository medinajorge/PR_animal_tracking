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
overwrite = False
mpl_val = False
density = 'pchip'

rho = False
optimize_spread = False
optimize_cmax = False

monotonic_q = False
cds = 'mercator'
exclude_me = False

output = 'models'

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:O:M:R:d:X:C:E:o:S:K:",
                               ["task=", "overwrite=", "mpl_val=", "rho=", "density=", "monotonic_q=", "cds=", "exclude_me=", "output=", "optimize_spread=", "optimize_cmax="])

except getopt.GetoptError:
    print('dist_best_models.py -t <task> -Q <s_q> -O <overwrite> -M <mpl_val> -R <rho> -d <density>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-t", "--task"):
        task = arg
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)
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
    elif opt in ("-o", "--output"):
        output = arg
    elif opt in ("-S", "--optimize_spread"):
        optimize_spread = fmt.decoder(arg)
    elif opt in ("-K", "--optimize_cmax"):
        optimize_cmax = fmt.decoder(arg)

phdu.getopt_printer(opts)
print(f"Output: {output}")

if output == 'models':
    input_kwargs = dict(task=task, overwrite=overwrite, mpl_val=mpl_val)
    if rho:
        input_kwargs['rho'] = rho
        if mpl_val:
            if optimize_spread:
                input_kwargs['optimize_spread'] = optimize_spread
            if optimize_cmax:
                input_kwargs['optimize_cmax'] = optimize_cmax
    if density != 'pchip':
        input_kwargs['density'] = density
    if cds != 'mercator':
        input_kwargs['cds'] = cds
    if monotonic_q:
        input_kwargs['monotonic_q'] = monotonic_q
    if exclude_me:
        input_kwargs['exclude_me'] = exclude_me

    print(f"input_kwargs:\n{input_kwargs}")

    _ = analysis.best_model_distribution(**input_kwargs)
elif output == 'pp':
    input_kwargs = dict(task=task, rho=rho)
    print(f"input_kwargs:\n{input_kwargs}")

    specs = analysis.dist_best_model_specs(**input_kwargs, mpl_val=True) # contains rho if passed
    del specs['dist_method']
    _ = df = analysis.pp_summary_dist(task=task, **specs, add_best_mode=True)
else:
    raise ValueError(f"Invalid output: {output}")

print("Done!")
