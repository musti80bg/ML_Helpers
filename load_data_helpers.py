import numpy as np

def load_house_data(path_to_data = "data"):
    full_path = f"./{path_to_data}/houses.txt"
    data = np.loadtxt(full_path, delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

