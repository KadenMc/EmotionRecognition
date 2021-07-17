import numpy as np
import pandas as pd


def load_csv_data(file, size=(48, 48)):
    df = pd.read_csv(file)
    y = df.emotion.values
    X = df.pixels.values
    X = np.array([np.reshape(np.array(x.split(' ')), size) for x in X.ravel()], dtype=float)
    print(X.shape)
    return X/255, y