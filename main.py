from flask import Flask, render_template, request
import json
import numpy as np
import pickle
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
from DiabetesLinearReg import regressor
local_server = True

with open("config.json", 'r') as c:
    params = json.load(c)['params']
model = pickle.load(open('model.pkl', 'rb'))

# if local_server:
#     app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
# else:
#     app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']
#
# db = SQLAlchemy(app)


# class Field(db.Model):
#     sno = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(80), nullable=False)
#     slug = db.Column(db.String(120), nullable=False)
#     content = db.Column(db.String(120), nullable=False)
#
#

@app.route('/')
def home():
    # field = Field.query.filter_by().all()
    return render_template("index.html")



@app.route('/predict', methods=["GET", "POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('diabetes.html', params=params,prediction_text='CO2    Emission of the vehicle is :{}'.format(output))

@app.route('/diabetes', methods=["GET", "POST"])
def diabetes():


    return render_template('diabetes.html', params=params)

@app.route('/sales', methods=["GET", "POST"])
def sales():

    int_features = [float(x) for x in request.form.values()]
    output=0

    return render_template('sales.html', params=params,prediction_text='CO2    Emission of the vehicle is :{}'.format(output))


@app.route('/about')
def about():
    return render_template('about.html', params=params)


if __name__ == '__main__':
    app.run(debug=True)


