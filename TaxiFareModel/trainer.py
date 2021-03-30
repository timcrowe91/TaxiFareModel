# imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, DistanceToCenterTransformer
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from sklearn.model_selection import train_test_split
from memoized_property import memoized_property
import pandas as pd
import numpy as np
import mlflow
from  mlflow.tracking import MlflowClient
import joblib



MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "timcrowe"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():
    def __init__(self, X, y, experiment_name, model):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = experiment_name
        self.model = model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        distance_pipeline = Pipeline([('DistanceTransformer', DistanceTransformer()), 
                              ('Scaler', RobustScaler())])
        time_pipeline = Pipeline([('TimeFeaturesEncoder', TimeFeaturesEncoder('pickup_datetime')), 
                            ('Encoder', OneHotEncoder(handle_unknown = 'ignore'))])
        distance_2_pipeline = Pipeline([('DistanceToCenterTransformer', DistanceToCenterTransformer()), 
                              ('Scaler', RobustScaler())])
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        dist_2_cols = ['dropoff_latitude', 'dropoff_longitude']

        preproc = ColumnTransformer([('distance', distance_pipeline, dist_cols),
                                    ('time', time_pipeline, time_cols),
                                    ('distance_center', distance_2_pipeline, dist_2_cols)])
        
        model_pipeline = Pipeline([('preproc', preproc),
                            ('regressor', self.model)])

        return model_pipeline

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()

        cross_score = cross_val_score(pipeline, self.X, self.y, cv=5, \
            scoring='neg_root_mean_squared_error').mean() * -1   
        self.mlflow_log_metric("cross val rmse", cross_score)

        pipeline.fit(self.X, self.y)
        self.pipeline = pipeline
        self.mlflow_log_param("model", str(model))
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        score = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("test rmse", score)
        return score

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, f'{str(self.model)}.joblib')





if __name__ == "__main__":
    models = [LinearRegression(), RandomForestRegressor(), KNeighborsRegressor(),\
        AdaBoostRegressor(), BaggingRegressor()]
    for model in models:
        print(f"Running for {model}")
        # get data
        #df = get_data(nrows = 1000)
        df = pd.read_csv("raw_data/train_10k.csv")
        # clean data
        df_cleaned = clean_data(df)
        # set X and y
        X = df_cleaned.drop(columns = 'fare_amount')
        y = df_cleaned['fare_amount']
        # hold out
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # train
        trainer = Trainer(X_train, y_train, EXPERIMENT_NAME, model)
        trainer.run()
        # evaluate
        trainer.evaluate(X_test, y_test)
        #score = trainer.evaluate(X_test, y_test)
        #print(score)
        trainer.save_model()
