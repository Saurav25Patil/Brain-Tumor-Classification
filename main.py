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

#Local Imports

from train_test_split import train_test_split_dataset
from load_images import load_dicom_image
from load_images import load_dicom_images_3d
from model import Trainer,Dataset, Model
from train import train_mri_type
from prediction import predict



data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
pytorch3dpath = "../input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"


mri_types = ['FLAIR','T1w','T1wCE','T2w']



sys.path.append(pytorch3dpath)
from efficientnet_pytorch_3d import EfficientNet3D

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(12)

#definition of main function
def main():
    # Sample loading of dicom images
    

    #Splitting training and test dataset
    df_train,df_valid=train_test_split_dataset()

    #Training the Dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelfiles = None

    if not modelfiles:
        modelfiles = [train_mri_type(df_train, df_valid, m) for m in mri_types]
        print(modelfiles)

    #Ensemble for Validation

    df_valid = df_valid.set_index("BraTS21ID")
    df_valid["MGMT_pred"] = 0
    for m, mtype in zip(modelfiles,  mri_types):
        pred = predict(m, df_valid, mtype, "train")
        df_valid["MGMT_pred"] += pred["MGMT_value"]
    df_valid["MGMT_pred"] /= len(modelfiles)
    auc = roc_auc_score(df_valid["MGMT_value"], df_valid["MGMT_pred"])
    print(f"Validation ensemble AUC: {auc:.4f}")
    sns.displot(df_valid["MGMT_pred"])

    #Ensemble for submission

    submission = pd.read_csv(f"{data_directory}/sample_submission.csv", index_col="BraTS21ID")

    submission["MGMT_value"] = 0
    for m, mtype in zip(modelfiles, mri_types):
        pred = predict(m, submission, mtype, split="test")
        submission["MGMT_value"] += pred["MGMT_value"]

    submission["MGMT_value"] /= len(modelfiles)
    submission["MGMT_value"].to_csv("submission.csv")

if __name__=="__main__":
    main()
