import numpy as np

def MSE(yPred, yTrue):
    error = yPred - yTrue
    print(f"{yPred} - {yTrue} = {error}")
    print(f"Error^2 = {error ** 2 }")
    return np.mean(error ** 2)