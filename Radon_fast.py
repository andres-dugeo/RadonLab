# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import math
import numpy as np
import datetime as dt
import collections
from helpers.TreeSet import TreeSet
from helpers.Max import Max

class Radon:

    def __init__(self, spatialDimensions, radonDimensions, sampleCount, sampleRateSeconds):
        self.spatialDimensions = spatialDimensions
        self.radonDimensions = radonDimensions
        self.sampleCount = sampleCount
        self.array = self._create_array(sampleCount, sampleRateSeconds)
        self.array_transpose = self._create_array_transpose(sampleCount, sampleRateSeconds) 
        self._norm_matrix = self._create_normalisation()

    def _create_array(self, sampleCount, sampleRateSeconds):
        def _get_indices(sd_index, rd_index, i):
            rd = self.radonDimensions[rd_index]
            sd = self.spatialDimensions[sd_index]
            index_delta = sd * rd * sampleRateSeconds
            index = i + index_delta
            lower_index = int(math.floor(index))
            upper_index = int(lower_index + 1)
            lower_fraction = 1 - index + lower_index
            upper_fraction = index - lower_index
            ret_array = []
            if lower_fraction != 0.0 and lower_index>=0 and lower_index < sampleCount:
                ret_array.append([lower_index, lower_fraction])
            if upper_fraction != 0.0 and upper_index>=0 and upper_index < sampleCount:
                ret_array.append([upper_index, upper_fraction])
            return ret_array
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        return [[[_get_indices(sd_index, rd_index , index) for sd_index in sd_range] for index in range(sampleCount)] for rd_index in rd_range]

    def _create_array_transpose(self, sampleCount, sampleRateSeconds):
        def _get_indices(sd_index, rd_index, i):
            rd = self.radonDimensions[rd_index]
            sd = self.spatialDimensions[sd_index]
            index_delta = sd * rd * sampleRateSeconds
            index = i - index_delta
            lower_index = int(math.floor(index))
            upper_index = int(lower_index + 1)
            lower_fraction = 1 - index + lower_index
            upper_fraction = index - lower_index
            ret_array = []
            if lower_fraction != 0.0 and lower_index>=0 and lower_index < sampleCount:
                ret_array.append([lower_index, lower_fraction])
            if upper_fraction != 0.0 and upper_index>=0 and upper_index < sampleCount:
                ret_array.append([upper_index, upper_fraction])
            return ret_array
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        return [[[_get_indices(sd_index, rd_index , index) for rd_index in rd_range] for index in range(sampleCount)] for sd_index in sd_range]

    def _create_normalisation(self):
        return_matrix = np.zeros((len(self.radonDimensions), self.sampleCount))
        compute = self._renormalisation(self._assign(return_matrix))
        rd_range = range(len(self.radonDimensions))
        for rd_index in rd_range:
            for index in range(self.sampleCount):
                compute(rd_index, index, 1)
        return return_matrix

    def _norm_test(self):
        return_matrix = np.zeros((len(self.radonDimensions), self.sampleCount))
        rd_range = range(len(self.radonDimensions))
        for rd_index in rd_range:
            for index in range(self.sampleCount):
                data= np.zeros((len(self.radonDimensions), self.sampleCount))
                compute = self._adjoint_inverse(self._forward(self._assign(data)))
                compute(rd_index, index, 1)
                return_matrix[rd_index][index]=data[rd_index][index]
        return return_matrix


    def forward(self, input):
        def _get_sum(rd_index, index):
            subarray = self.array[rd_index][index]
            return sum([sum([input[sd_index][indeces[0]]*indeces[1] for indeces in subarray[sd_index]]) for sd_index in sd_range])
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        sampleCount = input.shape[1]
        return np.array([[_get_sum(rd_index, index) for index in range(sampleCount)] for rd_index in rd_range])

    def adjoint_inverse(self, input):
        def _get_sum(sd_index, index):
            subarray = self.array_transpose[sd_index][index]
            return sum([sum([input[rd_index][indeces[0]]*indeces[1] for indeces in subarray[rd_index]]) for rd_index in rd_range])
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        sampleCount = input.shape[1]
        return np.array([[_get_sum(sd_index, index) for index in range(sampleCount)] for sd_index in sd_range])

    def _test_func(self, input):

        treeSet = TreeSet()
        radonData = self.forward(input)
        return_array = np.zeros((len(self.radonDimensions), self.sampleCount))
        index_range = range(self.sampleCount)

        for iter in range(5):

            needUpdate = [False for i in index_range]

            for index in index_range:
                max = self._getMax(radonData, index) 
                treeSet.add(max)
            max = treeSet.pollLast()

            while (max is not None):

                value = max.value / self._norm_matrix[max.firstIndex][max.secondIndex]
                return_array[max.firstIndex][max.secondIndex]  += value
                compute = self._adjoint_inverse(self._forward(self._update(radonData, needUpdate)))
                compute(max.firstIndex, max.secondIndex, value)
                if(iter<4):
                    if(max.secondIndex-1 >= 0):
                        compute(max.firstIndex, max.secondIndex-1, value/2.0)
                        return_array[max.firstIndex][max.secondIndex-1]  += value/2
                    if(max.secondIndex+1 < self.sampleCount):
                        compute(max.firstIndex, max.secondIndex+1, value/2.0)
                        return_array[max.firstIndex][max.secondIndex+1]  += value/2

                for index in index_range:
                    if (needUpdate[index]):
                        needUpdate[index] = False
                        max = self._getMax(radonData, index)
                        if (treeSet.remove(max)):
                            treeSet.add(max)

                max = treeSet.pollLast()

        return return_array

    def _adjoint_inverse(self, func):
        def _compute(rd_index, index, value):
            sd_range = range(len(self.spatialDimensions))
            for sd_index in sd_range:
                indices = self.array[rd_index][index][sd_index]
                for index_and_factor in indices:
                    func(sd_index, index_and_factor[0], index_and_factor[1] * value)
        
        return _compute

    def _forward(self, func):
        def _compute(sd_index, index, value):
            rd_range = range(len(self.radonDimensions))
            for rd_index in rd_range:
                indices = self.array_transpose[sd_index][index][rd_index]
                for index_and_factor in indices:
                    func(rd_index, index_and_factor[0], index_and_factor[1] * value)
        
        return _compute

    def _renormalisation(self, func):
        def _compute(rd_index, index, value):
            sd_range = range(len(self.spatialDimensions))
            for sd_index in sd_range:
                indices = self.array[rd_index][index][sd_index]
                for index_and_factor in indices:
                    func(rd_index, index, index_and_factor[1] * index_and_factor[1] * value)
        
        return _compute


    def _update(self, array, update):
        def _compute(d_index, index, value):
            array[d_index][index] -= value
            update[index] = True

        return _compute

    def _assign(self, array):
        def _compute(d_index, index, value):
            array[d_index][index] += value

        return _compute

  

    def _getMax(self, data, secondIndex):
        max = 0
        index = 0
        for firstIndex in range(len(data)):
            value = abs(data[firstIndex][secondIndex])
            if (value > max):
                max = value
                index = firstIndex

        return Max(index, secondIndex, data[index][secondIndex])

    
