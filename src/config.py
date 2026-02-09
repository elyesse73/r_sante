import os
import torch
import random
import numpy as np

CONFIG = {
    'MODE': 'NMF',  # 'AMF' (Atypical) or 'NMF' (Normal)
    'MODEL_TYPE': 'convnext_tiny', # 'tiny', 'small', 'base'
    'BATCH_SIZE': 32,
    'LR': 1e-4,
    'USE_MACENKO': True,
    'USE_SAMPLER': True,  # Essential for AMF and/or rare classes
    'NUM_WORKERS': 2,
    'TRAIN': 'full',  # 'full', 'feature_extraction' or 'fine_tune'
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Automatic path deduction based on mode
if CONFIG['MODE'] == 'AMF':
    ROOT_DIR = 'data/patches/atypical'
    IS_ATYPICAL = True
else:
    ROOT_DIR = 'data/patches/normal'
    IS_ATYPICAL = False

# Fix the seed everywhere for reproducibility (Essential in research)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True