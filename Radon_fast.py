# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import math
import numpy as np
import datetime as dt
import collections

class Radon_fast:

    def __init__(self, spatialDimensions, radonDimensions, sampleCount, sampleRateSeconds):
        self.spatialDimensions = spatialDimensions
        self.radonDimensions = radonDimensions
        self.array = self._create_array(sampleCount, sampleRateSeconds)
        self.array_transpose = self._create_array_transpose(sampleCount, sampleRateSeconds) 

    def _create_array(self, sampleCount, sampleRateSeconds):
        def _get_indices(sd_index, rd_index, i):
            rd = self.radonDimensions[rd_index]
            sd = self.spatialDimensions[sd_index]
            index_delta = sd * rd * sampleRateSeconds
            index = i + index_delta
            lower_index = math.floor(index)
            upper_index = lower_index + 1
            lower_fraction = 1 - index + lower_index
            upper_fraction = index - lower_index
            ret_array = []
            if lower_index>=0 and lower_index < sampleCount:
                ret_array.append([lower_index, lower_fraction])
            if upper_index>=0 and upper_index < sampleCount:
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
            lower_index = math.floor(index)
            upper_index = lower_index + 1
            lower_fraction = 1 - index + lower_index
            upper_fraction = index - lower_index
            ret_array = []
            if lower_index>=0 and lower_index < sampleCount:
                ret_array.append([lower_index, lower_fraction])
            if upper_index>=0 and upper_index < sampleCount:
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
