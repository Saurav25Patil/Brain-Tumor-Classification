import sys 
import json
import glob
import random
import collections
import time
import re
import os

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils import data as torch_data
from sklearn import model_selection as sk_model_selection
from torch.nn import functional as torch_functional
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
pytorch3dpath = "../input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"


def train_test_split_dataset():
    train_df = pd.read_csv(f"{data_directory}/train_labels.csv")

    df_train, df_valid = sk_model_selection.train_test_split(
        train_df,
        test_size=0.2,
        random_state=12,
        stratify=train_df["MGMT_value"],
    )

    return df_train,df_valid
