import numpy as np
import matplotlib.pyplot as plt

def LinearPlot(X, Y, weight, bias, width=7, height=4):
    plt.figure(figsize=(width, height))
    yp = weight * X + bias
    plt.plot(X, yp, label=f'(weight={weight:.3f}, bias={bias:.3f})', color='red',  linewidth=2)
    plt.scatter(X, Y)
    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def LogisticPlot(x, weight, bias, width=7, height=4):
    y = 1 / (1 + np.exp(-(weight * x + bias)))
    plt.figure(figsize=(width, height))
    plt.plot(x, y, label=f'(weight={weight:.3f}, bias={bias:.3f})', color='blue', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=-bias/weight, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
    plt.title('Sigmoid Curve')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()