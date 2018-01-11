"""
Aggregation functions that can be used as arguments to the Field.apply method
"""
from numba import jit
import numpy as np


def mask(values, above=None, below=None):
	"""Mask values not between above and below, in other words
	mask values below above and above below, its confusing but makes sense!
	"""

	if above != None or below != None:

		if above != None:
			values = np.ma.masked_less(values, float(above))

		if below != None:
			values = np.ma.masked_greater(values, float(below))

		return values

	else:
		return np.ma.masked_array(values)



def generic(func, values, axis=0, above=None, below=None):

	values = mask(values, above=above, below=below)

	return func(values, axis=axis)


def total(values, **kwargs):
	return generic(np.ma.sum, values, **kwargs)

def mean(values, **kwargs):
	return generic(np.ma.mean, values, **kwargs)

def max(values, **kwargs):
	return generic(np.ma.max, values, **kwargs)

def min(values, **kwargs):
	return generic(np.ma.min, values, **kwargs)


def count(values, **kwargs):
	return generic(np.ma.count, values, **kwargs)


def maxrun(values, axis=0, above=None, below=None, **kwargs):
	"""
	Maximum sequence length of non-masked values
	"""

	# Mask values
	values = mask(values, above=above, below=below)

	#Inverse of mask as 1s and 0s
	ones = (~np.ma.getmaskarray(values)).astype(np.int8)

	# Calculate cumulative sums of values in ones along axis resetting at every occurance of zero
	runs = np.zeros(ones.shape, dtype=np.int32)


	# Setup base slices
	s1 = [slice(0,None)] * len(ones.shape)
	s2 = [slice(0,None)] * len(ones.shape)

	# Set axis dimension slice to zero
	s1[axis] = 0
	runs[s1] = ones[s1]

	for i in range(1, ones.shape[axis]):

		# Set axis dimension slice indices
		s1[axis] = i - 1
		s2[axis] = i

		# Calculate the cumulative sum but multiply by the value,
		# this resets the sum to zero for every occurance of zero in ones
		runs[s2] = (runs[s1] + ones[s2]) * ones[s2]

	# Return the maximum run value
	return runs.max(axis=axis)




