import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models/genetic_ancestry_classification.pickle', 'rb'))

@app.route('/')
def home1():
    return render_template('geneticmodelindex.html')



@app.route('/predict', methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template('geneticmodelindex.html', prediction_text="ancestry is {}".format(prediction))


if __name__ == '__main__':
    app.run()

