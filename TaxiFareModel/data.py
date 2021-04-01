import pandas as pd
from google.cloud import storage

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=1000):
    # '''returns a DataFrame with nrows from s3 bucket'''
    # df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    # return df
    """method used in order to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv('gs://wagon-ml-crowe-le-wagon-project-309316/data/train_1k.csv', \
        nrows=nrows)
    return df.drop(df.columns[0], axis=1)


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()
