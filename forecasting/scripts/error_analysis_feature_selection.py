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
exclude_prediction_attrs = False
task = 'forecasting'
c = True
ef_abs_diff = None

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:e:T:c:E:",
                               ["target=", "exclude_prediction_attrs=", "task=", "c=", "ef_abs_diff="])
except getopt.GetoptError:
    print('error_analysis_feature_selection.py -t <target> -e <exclude_prediction_attrs> -T <task>')
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-t", "--target"):
        target = arg
    elif opt in ("-e", "--exclude_prediction_attrs"):
        exclude_prediction_attrs = fmt.decoder(arg)
    elif opt in ("-T", "--task"):
        task = arg
    elif opt in ("-c", "--c"):
        c = fmt.decoder(arg)
    elif opt in ("-E", "--ef_abs_diff"):
        ef_abs_diff = fmt.decoder(arg)

kwargs = dict(target=target, exclude_prediction_attrs=exclude_prediction_attrs, task=task, c=c, ef_abs_diff=ef_abs_diff)
print(f"kwargs:\n{kwargs}")
input_kwargs = kwargs.copy()
input_kwargs.update(params.TFT_specs[task])
if ef_abs_diff is None:
    del input_kwargs['ef_abs_diff']
print(f"input_kwargs:\n{input_kwargs}")

features = error_analysis.mrmr_selection(**input_kwargs)

print("Done!")
