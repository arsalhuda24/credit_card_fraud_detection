import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle, json
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_svm.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    with open("/Users/Arsal/examples/raltime_anomaly/test_df.json", 'r') as myfile:
        data = myfile.read()

    dat=json.loads(data)
    prediction = model.predict(pd.DataFrame.from_dict(dat[0].values()).T)
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

   # output = round(prediction[0], 2)
    output = prediction

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)