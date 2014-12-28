# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import math
import numpy as np
import datetime as dt
import collections

class Radon:

    def __init__(self, spatialDimensions, radonDimensions, sampleCount, sampleRateSeconds):
        self.spatialDimensions = spatialDimensions
        self.radonDimensions = radonDimensions
        self.sampleCount = sampleCount
        self.matrix = self._create_matrix(sampleCount, sampleRateSeconds)
        self.normalisingMatrix = self._create_normalisingMatrix(self.matrix)

    def _create_normalisingMatrix(self, matrix):
        transposed = matrix.transpose()
        product = np.dot(matrix, transposed)
        diagonal = np.diag(product)
        reciprocal = 1 / diagonal
        return np.diag(reciprocal)

    def _create_matrix(self, sampleCount, sampleRateSeconds):
        print "Start creating matrix, on",dt.datetime.now()
        ret_matrix = np.zeros((sampleCount * len(self.radonDimensions),sampleCount * len(self.spatialDimensions)))
        s_index = -1
        for sd in self.spatialDimensions:
            s_index+=1
            r_index = -1
            sdd = sd*sampleRateSeconds;
            for rd in self.radonDimensions:
                r_index+=1
                indexDelta = rd * sdd;
                for i in range(sampleCount):
                    index = i + indexDelta
                    lower_index = math.floor(index)
                    upper_index = lower_index + 1
                    lower_fraction = 1 - index + lower_index
                    upper_fraction = index - lower_index
                    if lower_index>=0 and lower_index < sampleCount:
                        ret_matrix[r_index * sampleCount + i][s_index * sampleCount + lower_index] += lower_fraction
                    if upper_index>=0 and upper_index < sampleCount:
                        ret_matrix[r_index * sampleCount + i][s_index * sampleCount + upper_index] += upper_fraction
        print "End creating matrix, on",dt.datetime.now()
        print "Rank:",np.linalg.matrix_rank(ret_matrix)
        return ret_matrix

    def forward(self, input):
        shape = (len(self.radonDimensions), input.shape[1])
        size = input.size
        return np.dot(self.matrix,input.reshape(size,1)).reshape(shape)

    def inverse(self, input):
        shape = input.shape
        size = input.size
        return np.linalg.solve(self.matrix,input.reshape(size,1)).reshape(shape)

    def adjointForward(self, input):
        shape = (len(self.radonDimensions), input.shape[1])
        size = input.size
        return np.linalg.lstsq(self.matrix.transpose(),input.reshape(size,1))[0].reshape(shape)

    def adjointInverse(self, input):
        shape = (len(self.spatialDimensions), input.shape[1])
        size = input.size
        return np.dot(self.matrix.transpose(),input.reshape(size,1)).reshape(shape)

    def adjointInverse_fast(self, input):
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        return [_get_indices(sd_index, rd_index , index) for sd_index in sd_range for rd_index in rd_range for index in range(sampleCount)]

    def newScheme(self, input, iterations): 
        shape_radon = (len(self.radonDimensions), input.shape[1])
        shape_spatial = (len(self.spatialDimensions), input.shape[1])
        size = input.size
        normalisedTranspose = np.dot(self.matrix.transpose(), self.normalisingMatrix)
        initial = np.dot(self.matrix, input.reshape(size,1))
        evaluator = np.dot(self.matrix, normalisedTranspose)

        temp = np.zeros_like(initial)  
        for i in xrange(0, iterations):
            mat = initial - np.dot(evaluator, temp)
            largest_index = Radon.argAbsMax(mat)
            largest_value = mat[largest_index]
            print largest_index, np.linalg.norm(mat,2), np.linalg.norm(mat,1), abs(largest_value)
            temp[largest_index] += largest_value
            
        retTuple = collections.namedtuple('matrices',['matrix_1','matrix_2'])    
        return retTuple(np.dot(self.normalisingMatrix,temp).reshape(shape_radon), np.dot(normalisedTranspose,temp).reshape(shape_spatial)) 


    @staticmethod
    def argAbsMax(mat):
        ind_max = np.unravel_index(mat.argmax(),mat.shape)
        ind_min = np.unravel_index(mat.argmin(),mat.shape)
        return ind_max if abs(mat[ind_max]) >= abs(mat[ind_min]) else ind_min
