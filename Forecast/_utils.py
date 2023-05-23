import pandas as pd
import numpy as np
from datetime import date, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns


"""
csv_folder: your csv folder
flist: your file list of csv files name (e.g. 467410-2016-01-01.csv)
start_date: start date of your data (e.g. 2016-01-01), default is min (2016-01-01)
end_date: end date of your data (e.g. 2016-12-31), default is None (to the last file).
"""
def create_dataset(csv_folder, flist, start_date = None, end_date = None):
    try:
        # cat all csvs
        df_list = pd.DataFrame()
        
        start_date = (date.fromisoformat(start_date) if start_date is not None else None)
        end_date = (date.fromisoformat(end_date) if end_date is not None else None)
        
        for i in range(len(flist)):
            this_date = date.fromisoformat(flist['filename'][i].split('.')[0][7:])
            if start_date is not None:
                if this_date < start_date:
                    continue

            if end_date is not None:
                if this_date > end_date:
                    continue
            tmp_df = pd.read_csv(csv_folder+flist['filename'][i], header=None)
            tmp_df = tmp_df.drop([0,1])
            tmp_df.insert(0, 'date', flist['filename'][i].split('.')[0][7:])
            df_list = pd.concat([df_list, tmp_df], axis=0)
        
        df_list = df_list.reset_index(drop=True)
        df_list.columns = ["date", "ObsTime","StnPres","SeaPres","Temperature","Td dew point","RH","WS","WD","WSGust","WDGust","Precp","PrecpHour","SunShine","GloblRad","Visb","UVI","Cloud Amount"]

    
    except Exception:
        raise Exception("Incorrect period! Except date from {} to {}, but got {} to {}".format(date.fromisoformat(df_list['date'][0]), date.fromisoformat(df_list['date'][len(df_list)-1]), start_date, end_date))

    return df_list


"""
dataset: your dataset
"""
def data_preproc(dataset):
    
    dataset["SunShine"] = dataset["SunShine"].replace('...', 0.0)
    dataset = dataset.replace('...', np.nan)
    dataset = dataset.replace('T', 0.0)
    dataset = dataset.replace('X', np.nan)
    dataset = dataset.replace('V', 0.0)
    dataset = dataset.replace('/', np.nan)
    

    # interpolate and round
    dataset[dataset.columns[1:]] = dataset[dataset.columns[1:]].astype(float)
    dataset = dataset.interpolate(method='linear', limit_direction='both', axis=0)
    dataset[dataset.columns[1:]] = dataset[dataset.columns[1:]].round(2)
    
    # add month and day
    dataset['month'] = dataset['date'].apply(lambda x: int(x.split('-')[1]))
    # dataset['month_onehot'] = dataset['month'].apply(lambda x: [1 if i == x else 0 for i in range(1,13)])
    dataset['day'] = dataset['date'].apply(lambda x: int(x.split('-')[2]))
    return dataset


"""
df: dataframe
cols: columns to plot
ticks: xticks interval (unit: day)
"""
def plot_time_series(df, cols, ticks = 7):
    # plot using sns
    sns.set(rc={'figure.figsize':(11, 4)})
    for col in cols:
        sns.lineplot(x="date", y=col, data=df)
        plt.xticks(df['date'][::ticks], rotation=90)

    
    plt.show()

"""
dataset: your dataset
step: how many days to predict next 1 day
dest_col: which column to predict, default is None (predict all columns)
return: (np.array(X), np.array(Y)) X: (n, 24, 17), Y: (n, 1, 17)
"""
def step_window(dataset, step=24, dest_col = None):
    X = []
    Y = []
    for i in range(0, len(dataset)-step, step):
        x_tmp = dataset[i:i+step]
        if dest_col is None:
            y_tmp = dataset[i+step:i+step+1]
        else:
            y_tmp = dataset[dest_col][i+step:i+step+1]
        
        X.append(x_tmp)
        Y.append(y_tmp)

    return np.array(X), np.array(Y)