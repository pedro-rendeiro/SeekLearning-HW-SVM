from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data = np.genfromtxt(
    "./datasets/dataset_train.txt",
    delimiter=",",
)
y = np.ravel(data[:, 2:], order="C")
print(
    pd.DataFrame(data=data, columns=["x1", "x2", "y"]).join(pd.DataFrame({"y_pred": y}))
)

clf_perceptron = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1),
    verbose=True,
)

print(clf_perceptron.get_params())
