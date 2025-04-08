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
from forecasting import ssm
from tidypath import fmt
import phdu
import getopt

# defaults
model = 'rw'
optimize_var = False
separate_mpl = False
num_trials = 500
recompute_mercator=False
task = 'forecasting'

try:
    opts, args = getopt.getopt(sys.argv[1:], "m:o:s:n:r:t:",
                               ["model=", "optimize_var=", "separate_mpl=", "num_trials=", "recompute_mercator=", "task="])
except getopt.GetoptError:
    print('ssm_val_fit.py -m <model> -o <optimize_var> -s <separate_mpl> -n <n>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-m", "--model"):
        model = arg
    elif opt in ("-o", "--optimize_var"):
        optimize_var = fmt.decoder(arg)
    elif opt in ("-s", "--separate_mpl"):
        separate_mpl = fmt.decoder(arg)
    elif opt in ("-n", "--num_trials"):
        num_trials = int(arg)
    elif opt in ("-r", "--recompute_mercator"):
        recompute_mercator = fmt.decoder(arg)
    elif opt in ("-t", "--task"):
        task = arg

phdu.getopt_printer(opts)
kwargs = dict(model=model, optimize_var=optimize_var, separate_mpl=separate_mpl, num_trials=num_trials, recompute_mercator=recompute_mercator, task=task)
print(f"kwargs:\n{kwargs}")

df = ssm.se_mpl_val(**kwargs)

print("Done!")
