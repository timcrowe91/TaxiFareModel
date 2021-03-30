from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        assert isinstance(X, pd.DataFrame)
        X_temp = X.copy()
        
        X_temp.index = pd.to_datetime(X_temp[self.time_column])
        X_temp.index = X_temp.index.tz_convert(self.time_zone_name)
        
        X_temp["dow"] = X_temp.index.weekday
        X_temp["hour"] = X_temp.index.hour
        X_temp["month"] = X_temp.index.month
        X_temp["year"] = X_temp.index.year
        
        return X_temp[['dow', 'hour', 'month', 'year']]



class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        assert isinstance(X, pd.DataFrame)
        X_temp = X.copy()
        X_temp['distance'] = haversine_vectorized(X_temp, 
                                                 start_lat = self.start_lat,
                                                 start_lon = self.start_lon,
                                                 end_lat = self.end_lat,
                                                 end_lon = self.end_lon)
        return X_temp[['distance']]


class DistanceToCenterTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self, 
                 nyc_lat="nyc_lat",
                 nyc_lon="nyc_lon", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.nyc_lat = nyc_lat
        self.nyc_lon = nyc_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        assert isinstance(X, pd.DataFrame)
        X_temp = X.copy()
        X_temp['nyc_lat'] = 40.7141667
        X_temp['nyc_lon'] = -74.0063889
        X_temp['distance_to_center'] = haversine_vectorized(X_temp, 
                                                 start_lat = self.nyc_lat,
                                                 start_lon = self.nyc_lon,
                                                 end_lat = self.end_lat,
                                                 end_lon = self.end_lon)
        return X_temp[['distance_to_center']]