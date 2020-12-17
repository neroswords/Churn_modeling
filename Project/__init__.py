from flask import Flask, request, abort, render_template
import json
import pickle
import Project.Config.predict

# models = []
# with open("models.pckl", "rb") as f:
#     while True:
#         try:
#             models.append(pickle.load(f))
#             print(models)
#         except EOFError:
#             break
app = Flask(__name__)
dt_model=pickle.load(open('./Project/Config/dt_model.pkl','rb'))
mlp_model=pickle.load(open('./Project/Config/mlp_model.pkl','rb'))
knn_model=pickle.load(open('./Project/Config/knn_model.pkl','rb'))
nb_model=pickle.load(open('./Project/Config/nb_model.pkl','rb'))
rfc_model=pickle.load(open('./Project/Config/rfc_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('landing.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    stay_in = 0
    leave = 0
    customer_name = request.form['id']
    customer_surname = request.form['lastname']
    customer_age = request.form['age']
    customer_gender = request.form['gender']
    customer_geography = request.form['geography']
    customer_score = request.form['score']
    customer_balance = request.form['balance']
    customer_credit_card = request.form['credit_card']
    customer_tenure = request.form['tenure']
    customer_active = request.form['active']
    customer_product = request.form['product']
    customer_salary = request.form['salary']
    features=[float(customer_score),
                float(customer_geography),
                float(customer_gender),
                float(customer_age),
                float(customer_tenure),
                float(customer_balance),
                float(customer_product),
                float(customer_credit_card),
                float(customer_active),
                float(customer_salary)]
    predict_data_dt = dt_model.predict([features])
    predict_data_mlp = mlp_model.predict([features])
    predict_data_knn = knn_model.predict([features])
    predict_data_nb = nb_model.predict([features])
    predict_data_rfc = rfc_model.predict([features])
    predict_list = [predict_data_dt,predict_data_mlp,predict_data_knn,predict_data_nb,predict_data_rfc]
    f = open("./Project/Config/history.txt", "a")
    f.write(str(features))
    f.write("\n")
    f.close()
    # print(predict_data_dt)
    # print(predict_data_mlp)
    # print(predict_data_knn)
    # print(predict_data_nb)
    # print(predict_data_rfc)
    for x in predict_list:
        if x == [0]:
            stay_in += 1
        elif x == [1]:
            leave +=1
        else:
            raise EnvironmentError
    if stay_in > leave :
        print("Stay")
        return render_template('result.html',pred='stay')
    else :
        print("leave")
        return render_template('result.html',pred='leave')
    
    # for model in models:
    #     predict_data = model.predict([features])
    #     print("2")
    #     print(predict_data)
    return "ok"

    

    # if output>str(0.5):
    #     return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    # else:
    #     return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
@app.route('/history',methods=['POST','GET'])
def history():
    f = open("./Project/Config/history.txt", "r")
    a = f.read()
    f.close()
    return render_template('history.html',hist=a)

