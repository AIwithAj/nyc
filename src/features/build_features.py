# make_dataset.py
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)


#Function aiming at calculating distances from coordinates
def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371 #km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
#Function aiming at calculating the direction
def ft_degree(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371 #km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
#Add distance feature
def build_features(df,save_path):



    df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,
                                                 df['pickup_longitude'].values, 
                                                 df['dropoff_latitude'].values,
                                                 df['dropoff_longitude'].values)
    #Add direction feature
    df['direction'] = ft_degree(df['pickup_latitude'].values,
                                    df['pickup_longitude'].values,
                                        df['dropoff_latitude'].values,
                                    df['dropoff_longitude'].values)
    df = df[(df.distance < 200)]
    # Create speed feature
    df['speed'] = df.distance / df.trip_duration
    #Remove speed outliers
    df = df[(df.speed < 30)]
    df.drop(['speed'], axis=1, inplace=True)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path + '/feature_df.csv', index=False)
    return df
def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["build_features"]

    input_file = "/data/interim/processed.csv"
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    save_p= home_dir.as_posix() + '/data/interim/feature_df'
    data = load_data(data_path)
    data= build_features(data,save_p)

    train_data, test_data = split_data(data, params['test_split'], params['seed'])
    save_data(train_data, test_data, output_path)

if __name__ == "__main__":
    main()