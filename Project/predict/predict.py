import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
import pickle

warnings.filterwarnings("ignore")
models = []

data = pd.read_csv("churn.csv")
features = data[data.columns[data.columns!='Exited'] ]
target = data[data.columns[data.columns=='Exited'] ]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train_std, y_train)
# dt_predict = dt.predict()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_std, y_train)
# knn_pred = knn.predict(x_test_std)

rfc = RandomForestClassifier(n_estimators=15, max_depth=None,
    min_samples_split=2, random_state=0)
rfc.fit(x_train_std, y_train)

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=0)
mlp.fit(x_train_std, y_train)
# nn_predict = mlp.predict(x_test_std)

nb = GaussianNB()
nb.fit(x_train_std, y_train)
# NB_pred = NB.predict(x_test_std)

models.append(('DecisionTree', dt))
models.append(('KNeighbors', knn))
models.append(('RandomForest', rfc))
models.append(('MLP', mlp))
models.append(('NavieBayes', nb))

# b = log_reg.predict_proba(final)

with open("models.pckl", "wb") as f:
    for model in models:
         pickle.dump(model, f)
# pickle.dump(dt,open('dt_model.pkl','wb'))
# model=pickle.load(open('dt_model.pkl','rb'))
