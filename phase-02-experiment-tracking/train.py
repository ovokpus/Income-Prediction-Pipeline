import argparse
import os
import pickle
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,\
    roc_auc_score, accuracy_score, roc_curve, f1_score
