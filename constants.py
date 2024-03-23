import os 
from dataloaders.bbci_dataloader import *
from metrics import *
from dataloaders.physionet_dataloader import *
from tqdm.notebook import tqdm 
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from model import EEGTCNet 
from keras.optimizers.legacy import Adam 
from sklearn.metrics import accuracy_score
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from typing import List
from mne.io import BaseRaw
from mne.io import read_raw_edf, concatenate_raws
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from conformal.conformal import * 
from conformal.conformal_metrics import *
from plotting import *

F1 = 8
KE = 32
KT = 4
L = 2
FT = 12
pe = 0.2
pt = 0.3

np.set_printoptions(suppress=True)