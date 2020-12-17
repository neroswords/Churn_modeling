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

data = pd.read_csv("./Project/Config/churn.csv")
features = data[data.columns[data.columns!='Exited'] ]
target = data[data.columns[data.columns=='Exited'] ]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std  = sc.transform(x_test)

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train_std, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_std, y_train)


rfc = RandomForestClassifier(n_estimators=15, max_depth=None,
    min_samples_split=2, random_state=0)
rfc.fit(x_train_std, y_train)

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,), random_state=0)
mlp.fit(x_train_std, y_train)

nb = GaussianNB()
nb.fit(x_train_std, y_train)
NB_pred = nb.predict(x_test_std)


pickle.dump(dt,open('Project/Config/dt_model.pkl','wb'))
pickle.dump(mlp,open('Project/Config/mlp_model.pkl','wb'))
pickle.dump(knn,open('Project/Config/knn_model.pkl','wb'))
pickle.dump(nb,open('Project/Config/nb_model.pkl','wb'))
pickle.dump(rfc,open('Project/Config/rfc_model.pkl','wb'))