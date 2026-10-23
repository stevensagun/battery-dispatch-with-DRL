import os
import sys
import json

from utils.utils import create_stats_file


# Build stats files for seeds that were trained as separate jobs.
# Usage: python -m main.make_stats log/de1_bat_hes/dqn/run/full [more paths ...]
for path in sys.argv[1:]:
    with open(os.path.join(path, 'inputs.json')) as f:
        exp_params = json.load(f)['EXP_PARAMS']
    create_stats_file(path, exp_params)
