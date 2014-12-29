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
        self.array = self._create_array(sampleCount, sampleRateSeconds)
        self.array_transpose = self._create_array_transpose(sampleCount, sampleRateSeconds) 

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

    def forward(self, input):
        def _get_sum(rd_index, index):
            subarray = self.array[rd_index][index]
            return sum([sum([input[sd_index][indeces[0]]*indeces[1]  for indeces in subarray[sd_index]]) for sd_index in sd_range])
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        sampleCount = input.shape[1]
        return np.array([[_get_sum(rd_index, index) for index in range(sampleCount)] for rd_index in rd_range])

    def adjoint_inverse(self, input):
        def _get_sum(sd_index, index):
            subarray = self.array_transpose[sd_index][index]
            return sum([sum([input[rd_index][indeces[0]]*indeces[1]  for indeces in subarray[rd_index]]) for rd_index in rd_range])
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        sampleCount = input.shape[1]
        return np.array([[_get_sum(sd_index, index) for index in range(sampleCount)] for sd_index in sd_range])

    def _test_func(self):
        def _get_mult(rd_index, rd2_index, index):
            ret_array = []
            subarray = self.array[rd_index][index]
            for sd_index in sd_range:
                indices = subarray[sd_index]
                for index_and_factor in indices:
                    index2 = index_and_factor[0]
                    factor = index_and_factor[1]
                    indices2 = self.array_transpose[sd_index][index2][rd2_index]
                    for index_and_factor_2 in indices2:
                        ret_array.append([index_and_factor_2[0], index_and_factor_2[1]*factor])
            return ret_array
        sd_range = range(len(self.spatialDimensions))
        rd_range = range(len(self.radonDimensions))
        sampleCount = self.sampleCount
        return [[[_get_mult(rd_index, rd2_index , index) for rd2_index in rd_range] for index in range(sampleCount)] for rd_index in rd_range]


