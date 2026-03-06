import deepchem as dc
import numpy as np

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
train, valid, test = datasets

X_train = train.X
y_train = train.y.flatten()

X_test = test.X
y_test = test.y.flatten()

print(f"Training samples: {X_train.shape[0]}")
print(f"Features per molecule: {X_train.shape[1]}")
print(f"Solubility range: {y_train.min():.2f} to {y_train.max():.2f}")
print("Data loaded successfully!")