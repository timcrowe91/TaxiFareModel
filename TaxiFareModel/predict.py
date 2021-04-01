import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.cloud import storage


PATH_TO_LOCAL_MODEL = 'model.joblib'
BUCKET_NAME = 'wagon-ml-crowe-le-wagon-project-309316'
AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


def get_test_data():
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    df = pd.read_csv('gs://wagon-ml-crowe-le-wagon-project-309316/data/test.csv')
    return df


# def get_model(path_to_joblib):
#     pipeline = joblib.load(path_to_joblib)
#     return pipeline


def download_model(bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    storage_location = 'models/taxi_fare_model/model.joblib'
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model



def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(kaggle_upload=False):
    df_test = get_test_data()
    pipeline = download_model(bucket=BUCKET_NAME) #joblib.load('model.joblib')
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':

    # ⚠ in order to push a submission to kaggle you need to use the WHOLE dataset
    nrows = 100
    generate_submission_csv(kaggle_upload=False)