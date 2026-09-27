"""Seed the live ImageGoal adapter without overriding official compilation."""
import os
import random
import runpy
import sys
import numpy as np
import torch

seed = int(os.environ.get('DIAGNOSTIC_SEED', '0'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
