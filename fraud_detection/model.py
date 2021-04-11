import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import OneClassSVM
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    path="/Library/Java/JavaVirtualMachines/Arsal/Dropbox/creditcard.csv"
    df = pd.read_csv(path)
    df.head()

    """Scaling
    1) We see that the Time and Amount are not scaled to other variables in the dataset
    2) create a balanced dataset 
    """
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df["scaled_amount"]=rob_scaler.fit_transform(df["Amount"].values.reshape(-1,1))
    df["scaled_time"]=rob_scaler.fit_transform(df["Time"].values.reshape(-1,1))

    # df.drop(['Time','Amount'], axis=1, inplace=True)

    tsne_data = df
    df2 = tsne_data[tsne_data.Class == 1]
    df2 = pd.concat([df2, tsne_data[tsne_data.Class == 0].sample(n = 280000)], axis = 0)
    df2_new=df2.drop(['Time','Amount'], axis=1)
    train, test = train_test_split(df2_new, test_size=.2)
    test1 = test.iloc[:,:-1]

    train_normal = train[train['Class']==0]
    train_normal = train_normal.iloc[:,:-1]
    train_outliers = train[train['Class']==1]
    train_outliers = train_outliers.iloc[:,:-1]
    outlier_prop = len(train_outliers) / len(train_normal)

    # model = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.000001)
    # model.fit(train_normal)

    with mlflow.start_run():
        svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.000001)
        svm.fit(train_normal)
        preds = svm.predict(test1)
        rmse, mae, r2 = eval_metrics(np.array(test.Class), preds)
        print("RMSE", rmse)
        print("MAE",  mae)
        print("r2", r2)
        #     mlflow.log_param("alpha", alpha)
        #     mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(svm, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(svm, "model")

    # filename = '/Users/Arsal/examples/fraud_detection/model_svm_mlflow.pkl'
    # test = "/Users/Arsal/examples/raltime_anomaly/test.pkl"
    # with open(filename, "wb+") as f:
    #     pickle.dump(svm, f)

# with open(test, "wb+") as g:
#     pickle.dump(inliers_test, g)
