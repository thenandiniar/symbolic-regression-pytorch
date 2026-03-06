import torch
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors
from expression import Operator, ExpressionTree
from fitness import FitnessFunction
from genetic_algo import SymbolicRegressor

print("Loading Delaney dataset...")
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
train, valid, test = datasets

def compute_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        desc = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
        descriptors.append(desc)
    return np.array(descriptors)

print("Computing molecular descriptors...")
X_train = compute_descriptors(train.ids)
y_train = train.y.flatten()

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

print(f"Data ready: {X_tensor.shape}")
print("Features: LogP, MolWt, HBD, HBA, TPSA, RotBonds")
print()

print("Running Symbolic Regression...")
regressor = SymbolicRegressor(
    population_size=200,
    max_depth=3,
    operators=[Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV]
)

best_tree, history = regressor.evolve(
    X_tensor, y_tensor,
    generations=100,
    verbose=True
)

print()
print("Best equation found:")
print(best_tree)

# Calculate RMSE
X_test_desc = compute_descriptors(test.ids)
X_test_tensor = torch.tensor(X_test_desc, dtype=torch.float32)
y_test = test.y.flatten()
y_pred = best_tree.evaluate(X_test_tensor).detach().numpy()
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f"Test RMSE: {rmse:.3f}")