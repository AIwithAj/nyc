# make_dataset.py
import pathlib
import yaml
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def preprocess(df):
    df = df[(df.trip_duration < 5900)]
    #Only keep trips with passengers
    df = df[(df.passenger_count > 0)]

    df = df[(df.pickup_longitude > -100)]
    df = df[(df.pickup_latitude < 50)]
    df['trip_duration'] = np.log(df['trip_duration'].values)
    #One-hot encoding binary categorical features
    df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'])], axis=1)

    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['vendor_id'])], axis=1)
    df.drop(['vendor_id'], axis=1, inplace=True)
    #Datetyping the dates
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)

    df.drop(['dropoff_datetime'], axis=1, inplace=True) #as we don't have this feature in the testset

    #Date features creations and deletions
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    df.drop(['minute'], axis=1, inplace=True)
    df.drop(['pickup_datetime'], axis=1, inplace=True)
    return df
def save_data(data, output_path):
    # Save the preprocess datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path + '/processed.csv', index=False)
    
def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    input_file = "/data/raw/train.csv"
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/interim'
    data = load_data(data_path)
    preprocess_df=  preprocess(data)
    save_data(preprocess_df,output_path)

if __name__ == "__main__":
    main()