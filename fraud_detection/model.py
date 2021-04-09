import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('/Users/Arsal/GRIP/creditcard.csv')
df.head()

inliers = df[df.Class==0]
inliers = inliers.drop(['Class'], axis=1)
outliers = df[df.Class==1]
outliers = outliers.drop(['Class'], axis=1)
inliers_train, inliers_test = train_test_split(inliers, test_size=0.30, random_state=42)

model = IsolationForest()
model.fit(inliers_train)
inlier_pred_test = model.predict(inliers_test)
outlier_pred = model.predict(outliers)
print(outlier_pred)

filename = '/Users/Arsal/examples/raltime_anomaly/model.pkl'
test = "/Users/Arsal/examples/raltime_anomaly/test.pkl"
with open(filename, "wb+") as f:
    pickle.dump(model, f)

with open(test, "wb+") as g:
    pickle.dump(inliers_test, g)
