import netCDF4
import numpy as np
from collections import OrderedDict



def generic(values, keyfunc):

	result = OrderedDict()

	# Step through all values
	for index in range(0, len(values)):

		# Generate key
		key = keyfunc(values[index])

		# Add to results dictionary
		if key not in result.keys():
			result[key] = [[index]]
		else:
			result[key][0].append(index)

	return result, None



def yearmonth(values, bounds=False):

	def keyfunc(value):
		return value.year, value.month

	return generic(values, keyfunc)


def month(values, bounds=False):

	def keyfunc(value):
		return value.month

	return generic(values, keyfunc)


def year(values, bounds=False):

	def keyfunc(value):
		return value.year

	return generic(values, keyfunc)


def geometry(source, target=None):

	result = OrderedDict()

	for i in range(len(target)):

		result[i] = [slice(None)] * len(source.shape)

	print(result[0])

	return result, None
