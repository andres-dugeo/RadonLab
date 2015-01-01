# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import Radon as RadonBase
import math
import numpy as np
import datetime as dt
import collections

class Radon(RadonBase.Radon):

    def __init__(self, spatialDimensions, weights, radonDimensions, sampleCount, sampleRateSeconds):
        self.weightMatrix = self._create_weightMatrix(weights, sampleCount) 
        RadonBase.Radon.__init__(self, spatialDimensions, radonDimensions, sampleCount, sampleRateSeconds) 

    def _create_normalisingMatrix(self, matrix):
        transposed = matrix.transpose()
        print matrix.shape, self.weightMatrix.shape, transposed.shape
        product = np.dot(np.dot(matrix, self.weightMatrix), transposed)
        diagonal = np.diag(product)
        reciprocal = 1 / diagonal
        return np.diag(reciprocal)

    def _create_weightMatrix(self, weights, sampleCount):
        diagonal = [weights[i/sampleCount] for i in xrange(len(weights)*sampleCount)]
        return np.diag(diagonal)

    def forward(self, input):
        shape = (len(self.radonDimensions), input.shape[1])
        size = input.size
        return np.dot(np.dot(self.matrix, self.weightMatrix),input.reshape(size,1)).reshape(shape)

    def inverse(self, input):
        shape = input.shape
        size = input.size
        return np.linalg.lstsq(np.dot(self.matrix, self.weightMatrix),input.reshape(size,1))[0].reshape(shape)



    def newScheme(self, input, iterations): 
        shape_radon = (len(self.radonDimensions), input.shape[1])
        shape_spatial = (len(self.spatialDimensions), input.shape[1])
        size = input.size
        normalisedTranspose = np.dot(self.matrix.transpose(), self.normalisingMatrix)
        initial = np.dot(np.dot(self.matrix, self.weightMatrix), input.reshape(size,1))
        evaluator = np.dot(np.dot(self.matrix, self.weightMatrix), normalisedTranspose)

        max_value = -1
        max_L1norm = -1
        max_L2norm = -1
        stats = []
        temp = np.zeros_like(initial)  
        for i in xrange(0, iterations):
            mat = initial - np.dot(evaluator, temp)
            largest_index = RadonBase.Radon.argAbsMax(mat)
            largest_value = mat[largest_index]
            l1norm = np.linalg.norm(mat,1)
            l2norm = np.linalg.norm(mat,2)
            max_value = max(abs(largest_value), max_value)
            max_L1norm = max(l1norm, max_L1norm)
            max_L2norm = max(l2norm, max_L2norm)
            stats.append([l1norm/max_L1norm, l2norm/max_L2norm, abs(largest_value)/max_value])
            temp[largest_index] += largest_value
            
        retTuple = collections.namedtuple('matrices',['matrix_1','matrix_2','matrix_3','stats'])
        return retTuple(np.dot(self.normalisingMatrix,temp).reshape(shape_radon), np.dot(normalisedTranspose,temp).reshape(shape_spatial), mat.reshape(shape_radon), stats) 



