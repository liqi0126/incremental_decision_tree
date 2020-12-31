import pandas as pd
import numpy as np

def init_csv(device, instrument, filename):
    df = pd.read_csv(filename, keep_default_na=False)
    df['Device'] = np.full(len(df), device)
    df['Instrument'] = np.full(len(df), instrument)
    return df

dfs = []
dfs.append(init_csv('phone', 'accelerometer', 'Phones_accelerometer.csv'))
dfs.append(init_csv('phone', 'gyroscope', 'Phones_gyroscope.csv'))
dfs.append(init_csv('watch', 'accelerometer', 'Watch_accelerometer.csv'))
dfs.append(init_csv('watch', 'gyroscope', 'Watch_gyroscope.csv'))

df = pd.concat(dfs)
df.to_csv('activity_recognition.csv', columns=['x','y','z','Device','Instrument','gt'], index=None)
