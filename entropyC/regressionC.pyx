import cython
import numpy as cnp
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef class LinearRegression:
    cdef public double bias
    cdef public double weight
    
    def __init__(self):
        self.bias = 0.0
        self.weight = 0.0

    def train(self, double[:, :] x, double[:, :] y, long long epoch, double lr):
        cdef int n = x.shape[0]
        cdef int m = x.shape[1]
        cdef int i, j, e
        cdef double dw, db, errorSum, errorXSum, yPred, error
        
        for e in range(epoch):
            errorSum = 0.0
            errorXSum = 0.0
            for i in range(n):
                for j in range(m):
                    yPred = (self.weight * x[i, j]) + self.bias
                    error = yPred - y[i, 0]
                    errorSum += error
                    errorXSum += error * x[i, j]
            
            dw = (2.0 / n) * errorXSum
            db = (2.0 / n) * errorSum
            self.weight -= lr * dw
            self.bias -= lr * db

    def predict(self, double[:] X):
        cdef int n = X.shape[0]
        nparr = cnp.zeros(n, dtype=cnp.float64)
        cdef double[:] yPred = nparr
        cdef int i
        for i in range(n):
            yPred[i] = (self.weight * X[i]) + self.bias
        return nparr

    
cdef class LinearROLS:
    cdef public double weight
    cdef public double bias
    
    def __init__(self):
        self.bias = 0.0
        self.weight = 0.0

    def train(self, double[:, :] X, double[:, :] Y):
        cdef double sigmaX, sigmaY, sigmaXS, sigmaXY
        cdef int i, n
        
        n = X.shape[0]
        m = X.shape[1]

        for i in range(n):
            for j in range(m):
                sigmaXS += X[i, j] ** 2
                sigmaX += X[i, j]
                sigmaY += Y[i, j]
                sigmaXY += X[i, j] * Y[i, 0]
                    
        self.weight = ((n * sigmaXY) - (sigmaX * sigmaY)) / ((n * sigmaXS) - sigmaX ** 2)
        self.bias = (sigmaY - (self.weight * sigmaX)) / n 

    def predict(self, double[:] X):
        cdef int n = X.shape[0]
        nparr = cnp.zeros(n, dtype=cnp.float64)
        cdef double[:] yPred = nparr
        cdef int i
        for i in range(n):
            yPred[i] = (self.weight * X[i]) + self.bias
        return nparr

class LogisticRegression:
    cdef public double bias
    cdef public double weight
    
    def __init__(self):
        self.bias = 0.0
        self.weight = 0.0

    def train(self, double[:, :] X, double[:, :] Y, int epoch, dobule lr):
        cdef int i, j, k
        cdef double z, yPred, error, dw, db
        
        n = X.shape[0]
        m = X.shape[1]
        o = Y.shape[1]
        
        for i in range(epoch):
            for i in range(n):
                for j in range(m):
                    z = self.weight * X[i, j] + self.bias
                    yPred = 1 / (1 + cnp.exp(-z))
                    error = yPred - Y[i, j]

            dw = np.sum(X * error) / n
            db = np.sum(error) / n

            self.weight -= lr * dw
            self.bias -= lr * db

    def predict(self, double[:] X):
        cdef int n = X.shape[0]
        nparr = cnp.zeros(n, dtype=cnp.float64)
        cdef double[:] yPred = nparr
        cdef int i
        for i in range(n):
            yPred[i] = (self.weight * X[i]) + self.bias
        return nparr