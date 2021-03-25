import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import sys

loaded_rf = joblib.load("./random_forest.joblib")
print( loaded_rf.predict([[ float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) ]] ) )




