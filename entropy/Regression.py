import pandas as pd
import numpy as np
from sympy import symbols, init_printing, latex
from IPython.display import display, Math

init_printing(use_latex='mathjax') 


class LogisticRegression:
    def __init__(self):
        self.bias = 0.0
        self.weight = 0.0

    def train(self, x, y, epoch, lr, tol):
        X = np.array(x).reshape(-1, 1)
        Y = np.array(y).reshape(-1, 1)
        m = len(X)
        self.P = lambda za: 1 / (1 + np.exp(-za))
        lossB = np.inf
        
        for i in range(epoch):
            z = self.weight * X + self.bias
            yPred = self.P(z)

            #lossN = -(1/m) * np.sum(Y * np.log(yPred) + (1 - Y) * np.log(1 - yPred))

            """if abs(lossB - lossN) < tol:
                print(f"Convergen at iteration-{i}")
                break""" #Not working properly so i comment this
            
            #lossB = lossN
            error = yPred - Y
            
            dw = np.sum(X * error) / m
            db = np.sum(error) / m

            self.weight -= lr * dw
            self.bias -= lr * db

        return self.weight, self.bias

    def predict(self, x):
        P = self.P(za=self.weight * x + self.bias) 
        return P


class LinearRegression:
    def __init__(self):
        self.bias = 0
        self.weight = 0

    def train(self, x, y, epoch, lr):
        X = np.array(x).reshape(-1, 1)
        Y = np.array(x).reshape(-1, 1)
        self.P = lambda w, b, x: (w * x) + b
        n = len(X)
        
        for i in range(epoch):
            yPred = self.P(self.weight, self.bias, X)
            error = yPred - Y
            
            dw = (2/n) * np.sum(error * X)
            db = (2/n) * np.sum(error)
            
            self.weight -= lr * dw
            self.bias -= lr * db
            mse = np.mean(error**2)

        return self.weight, self.bias
    
    def predict(self, X):
        return self.P(self.weight, self.bias, X)

    def showFormula(self):
        print("Model: ")
        display(Math(f'f(x) = {self.weight:.4f} \cdot x + {self.bias:.4f}'))
        print("Formulas: ")
        display(Math(r'f(x) = w \cdot x + b'))
        display(Math(r'dw = \frac{2}{n} \sum_{i=1}^{n} (y_{pred} - y_i) \cdot x_i'))
        display(Math(r'db = \frac{2}{n} \sum_{i=1}^{n} (y_{pred} - y_i)'))
        