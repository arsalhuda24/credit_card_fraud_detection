import pickle


with open("/Users/Arsal/examples/raltime_anomaly/model_svm.pkl", 'rb+') as f:
    model = pickle.load(f)

print(model)