from flask import Flask, request, abort, render_template
import json
import pickle
from Project.predict import *

models = []
with open("./Project/predict/models.pckl", "rb") as f:
    while True:
        try:
            models.append(pickle.load(f))
            # print(models)
        except EOFError:
            break
# client = pymongo.MongoClient("")
# db = client.test
# col = db["User"]

app = Flask(__name__)
# dt_model=pickle.load(open('dt_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    # print(int_features)
    # print(final)
    # prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")