import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import subprocess

# Do data processing step
subprocess.call(["python", "data_proc.py"])

# Feature engineering on Train.csv
subprocess.call(["python", "feature_eng.py", "train"])
# Save intermediate csvs in train/
# subprocess.call(["python", "feature_eng.py", "train", "save_intermediate"])

# Feature engineering on Valid.csv
subprocess.call(["python", "feature_eng.py", "valid"])
# Save intermediate csvs in valid/
# subprocess.call(["python", "feature_eng.py", "valid", "save_intermediate"])

# Feature engineering on Test.csv
subprocess.call(["python", "feature_eng.py", "test"])
# Save intermediate csvs in test/
# subprocess.call(["python", "feature_eng.py", "test", "save_intermediate"])

# Model and predict on Valid.csv, evaluate on ValidSolution.csv
subprocess.call(["python", "model.py", "valid"])

# Model and predict probabilities on Test.csv, prepare submission
subprocess.call(["python", "model.py", "test"])

