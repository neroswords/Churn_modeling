import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("Churn_Modelling.csv")
# data = np.array(data)

# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('int')
# X = X.astype('int')
dt = DecisionTreeClassifier(criterion="entropy")
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



dt.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

# b = log_reg.predict_proba(final)


pickle.dump(dt,open('dt_model.pkl','wb'))
model=pickle.load(open('dt_model.pkl','rb'))
