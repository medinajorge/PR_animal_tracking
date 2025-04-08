import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from inspect import signature
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
import getopt
from tidypath import fmt
import phdu

# Default params
params_idx = 0
fair = False
task = 'forecasting'
partition = 'test'
expand_encoder_until_future_length = True
store_missing_idxs = False
decoder_missing_zero_loss = False
predict_shift = None
max_train_days = 28

# Get params from command line args
try:
    opts, args = getopt.getopt(sys.argv[1:], "f:i:t:p:E:S:D:P:T:",
                               ["fair=", "params_idx=", "task=", "partition=", "expand_encoder_until_future_length=",
                                "store_missing_idxs=", "decoder_missing_zero_loss=", "predict_shift=", "max_train_days="])
except getopt.GetoptError:
    print('check errors')

for opt, arg in opts:
    if opt in ("-f", "--fair"):
        fair = fmt.decoder(arg)
    elif opt in ("-i", "--params_idx"):
        params_idx = int(arg)
    elif opt in ("-t", "--task"):
        task = arg
    elif opt in ("-p", "--partition"):
        partition = arg
    elif opt in ("-E", "--expand_encoder_until_future_length"):
        expand_encoder_until_future_length = fmt.decoder(arg)
    elif opt in ("-S", "--store_missing_idxs"):
        store_missing_idxs = fmt.decoder(arg)
    elif opt in ("-D", "--decoder_missing_zero_loss"):
        decoder_missing_zero_loss = fmt.decoder(arg)
    elif opt in ("-P", "--predict_shift"):
        predict_shift = fmt.decoder(arg)
    elif opt in ("-T", "--max_train_days"):
        max_train_days = int(arg)

phdu.getopt_printer(opts)

kws = dict(fair=fair, params_idx=params_idx, task=task, CI_mpl=False, partition=partition)
if store_missing_idxs:
    kws['store_missing_idxs'] = True
if max_train_days != 28:
    kws['max_train_days'] = max_train_days
if task == 'imputation':
    kws['epochs'] = 200
    if expand_encoder_until_future_length:
        kws['expand_encoder_until_future_length'] = True
    if predict_shift is not None:
        kws['predict_shift'] = predict_shift
print(f"Params:\n{kws}")
baseline_width = analysis.quantile_baseline_lengths(**kws)

print("Done!")
