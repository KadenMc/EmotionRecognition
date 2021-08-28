import numpy as np
import pandas as pd

RACE_MAP = {
    -1: "undefined",
    1: "unknown",
    2: "white",
    3: "black",
    4: "asian",
    5: "brown",
    6: "invalid"
}

RACE_MAP_INV = {v: k for k, v in RACE_MAP.items()}


def filter_data(condition_bool, arrs):
    # Get indices
    inds = np.argwhere(condition_bool)[:, 0]
    
    # Filter based on indices
    for i, arr in enumerate(arrs):
        arrs[i] = arr[inds]

    return arrs

"""# Filter based on indices
arr_new = []
for i, arr in enumerate(arrs):
    arr_new.append(arr[inds])

return arr_new"""


def get_races(X, y, race):
    race_data = dict()
    for k in RACE_MAP:
        race_data[RACE_MAP[k]] = filter_data(race == k, [X, y])

    return race_data

def load_csv(file, size=(48, 48)):
    # Load the file
    df = pd.read_csv(file)
    
    # Define the data
    X = df.pixels.values
    X = np.array([np.reshape(np.array(x.split(' ')), size) for x in X.ravel()], dtype=np.uint8)
    y = df.emotion.values
    race = df.Race.values

    print("Originally:", len(X))
    
    # Remove images with no race label
    X, y, race = filter_data(race != RACE_MAP_INV["undefined"], [X, y, race])
    
    print("With race:", len(X))
    
    # Remove invalid images
    X, y, race = filter_data(race != RACE_MAP_INV["invalid"], [X, y, race])
    
    print("Valid:", len(X))
    
    return X, y, race