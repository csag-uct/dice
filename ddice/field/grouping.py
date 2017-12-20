import netCDF4
from collections import OrderedDict




def generic(values, keyfunc):

	result = OrderedDict()

	# Step through all times
	for index in range(0, len(values)):

		# Generate key
		key = keyfunc(values[index])

		# Add to results dictionary
		if key not in result.keys():
			result[key] = [index]
		else:
			result[key].append(index)

	return result



def yearmonth(values):

	def keyfunc(value):
		return value.year, value.month

	return generic(values, keyfunc)


def month(values):

	def keyfunc(value):
		return value.month

	return generic(values, keyfunc)


def year(values):

	def keyfunc(value):
		return value.year

	return generic(values, keyfunc)
