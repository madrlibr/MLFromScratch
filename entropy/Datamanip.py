import pandas as pd

def splitData(x, y, test_size, random_state):
    n = int(len(x) * test_size)
    x_test = x.sample(frac=test_size, random_state=random_state)
    y_test = y.sample(frac=test_size, random_state=random_state)
    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)

    return x_train, y_train, x_test, y_test