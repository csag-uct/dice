import numpy as np

def real_slices(shape, slices=()):
	"""Turns a possibly truncated mixture of slices and integer indices into a full tuple with
	the same length as dimensionality, of valid array indices (slice, integer, or iterable)
	"""

	# If slices is not a list/tuple then make it one
	if not hasattr(slices, '__iter__'):
		slices = (slices,)

	# Setup a default set of slices based on the shape, this means that missing
	# slices will take on default values
	result = [slice(0,size) for size in list(shape)]

	# Process each dimension specified
	for i in range(len(slices)):

		s = slices[i]

		# Process actual slice instances converting Nones to values
		if isinstance(s, slice):

			start, stop, step = s.start, s.stop, s.step

			if start == None:
				start = 0
			if stop == None:
				stop = shape[i]
			if step == None:
				step = 1

			result[i] = slice(start, stop, step)

		# Turn single int values into slice equivalents
		elif isinstance(s, int):
			result[i] = slice(slices[i],slices[i]+1,1)

		# Iterables can be used directly
		elif hasattr(s, '__iter__'):
			result[i] = s

	return tuple(result)


def reslice(shape, slices, newslices):
	"""Takes a list of slices (or integers or iterables), an array shape, and applies new slices to them to produce
	a list of modified slices.  Returns a tuple of (shape, slices) where shape is the newly calculated shape

	>>> reslice((10,), (), (slice(3,-2),))
	((5,), (slice(3, 8, None),))
	>>> reslice((10,), (slice(3, 7, None),), ([0,2,3],))
	((3,), (array([3, 5, 6]),))
	>>> reslice((10,10), (slice(3, 7, None),), ([0,2,3],))
	((3, 10), (array([3, 5, 6]), slice(0, 10, None)))
	>>> reslice((10,10), (slice(None), slice(3, 7, None)), ([0,2,3],))
	((3, 10), (array([0, 2, 3]), slice(3, 7, 1)))
	>>> reslice((10,), (), (slice(2,-2),))
	((6,), (slice(2, 8, None),))
	"""

	# First get rationalized real slices
	slices = list(real_slices(shape, slices))

	# Default shape is the original shape
	newshape = list(shape)

	# Check that newslices is an iterable, if not, make it so
	if not hasattr(newslices, '__iter__'):
		newslices = (newslices,)

	for i in range(len(newslices)):

		s = slices[i]

		ns = newslices[i]

		was = None

		# Convert slice instances to arrays
		if isinstance(s, slice):
			was = slice

			start, stop, step = s.start, s.stop, s.step

			if not start:
				start = 0
			if not stop:
				stop = shape[i] + 1
			if not step:
				step = 1

			s = np.arange(start, stop, step)

		# Integer indices also become single element arrays
		elif isinstance(s, int):
			was = int
			s = np.array(s)

		# Iterables become arrays
		elif hasattr(s, '__iter__'):
			s = np.array(s)

		# Otherwise throw an error
		else:
			raise ArrayError('Invalid slice {}'.repr(s))

		# Reslice
		s = s[ns]

		# Set new dimension size
		if hasattr(s, '__iter__'):
			newshape[i] = len(s)
		else:
			newshape[i] = 1


		# Try and get back to the type of slice we had originally
		if was == slice  and isinstance(ns, slice):
			slices[i] = slice(s[0], s[-1]+1, ns.step)

		elif was == int:
			slices[i] = slice(s,s+1,1)

		elif isinstance(s, int):
			slices[i] = slice(s,s+1,1)

		else:
			slices[i] = s


	return tuple(newshape), tuple(slices)
