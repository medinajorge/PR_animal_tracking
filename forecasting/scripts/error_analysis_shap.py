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
from forecasting import error_analysis, params
from tidypath import fmt
import phdu
import getopt

# defaults
target = 'Q'
task = 'forecasting'
exclude_prediction_attrs = False
all_features = True
c = True
ef_abs_diff = None
overwrite = False

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:T:e:a:c:E:O:",
                               ["target=", "task=", "exclude_prediction_attrs=", "all_features=", "c=", "ef_abs_diff=", "overwrite="])

except getopt.GetoptError:
    print('error_analysis_hp.py -t <target> -T <task> -e <exclude_prediction_attrs> -a <all_features>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-t", "--target"):
        target = arg
    elif opt in ("-T", "--task"):
        task = arg
    elif opt in ("-e", "--exclude_prediction_attrs"):
        exclude_prediction_attrs = fmt.decoder(arg)
    elif opt in ("-a", "--all_features"):
        all_features = fmt.decoder(arg)
    elif opt in ("-c", "--c"):
        c = fmt.decoder(arg)
    elif opt in ("-E", "--ef_abs_diff"):
        ef_abs_diff = fmt.decoder(arg)
    elif opt in ("-O", "--overwrite"):
        overwrite = fmt.decoder(arg)

phdu.getopt_printer(opts)
kwargs = dict(target=target, task=task, exclude_prediction_attrs=exclude_prediction_attrs, all_features=all_features, c=c, use_tqdm=True, overwrite=overwrite, ef_abs_diff=ef_abs_diff)
print(f"kwargs:\n{kwargs}")
input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
if ef_abs_diff is None:
    del input_kwargs['ef_abs_diff']
print(f"input_kwargs:\n{input_kwargs}")

study = error_analysis.get_shap_values(**input_kwargs)

print("Done!")
