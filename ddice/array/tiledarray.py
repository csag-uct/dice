import numpy as np

from array import Array
from array import real_slices, reslice

from nparray import numpyArray



class tiledArray(Array):
	"""
	An Array implemention that uses numpy arrays to store subsets/tiles of the full data
	an supports sparse tile sets (some tiles can be missing) and parallel processing
	across tiles

	Tiles are defined by a list of length Dn where each list element is a list of tile break
	points on the corresponding array axis and Dn is the dimensionality of the array.
	"""


	def __init__(self, shape, dtype, tilespec=None, storage=numpyArray):
		"""
		Create an instance of a tiled Array.  Sets up the tile dictionary and tile bounds arrays
		but doesn't allocate and storage as allocation is lazy

		>>> a = tiledArray((100,16,20), dtype=np.float32, tilespec=[[20,40,60,80],[5,10,15],[5,10,15]])
		>>> print(a._tilespec)
		[[0, 20, 40, 60, 80, 100], [0, 5, 10, 15, 16], [0, 5, 10, 15, 20]]

		"""

		super(tiledArray, self).__init__(shape, dtype)

		# If no tilespec then (for now) we create one big tile
		if tilespec == None:
			self._tilespec = [[size] for size in shape]

		# Otherwise we make sure each set of bounds doesn't start at zero and ends at size
		else:

			self._tilespec = tilespec

			for d in range(len(shape)):

				if self._tilespec[d][0] < 0:
					self._tilespec[d] = 0

				if self._tilespec[d][0] > 0:
					self._tilespec[d].insert(0,0)

				if self._tilespec[d][-1] < shape[d]:
					self._tilespec[d].append(shape[d])

				if self._tilespec[d][-1] > shape[d]:
					self._tilespec[d][-1] = shape[d]


		self._tiles = self._make_tiles(self.shape, self._tilespec)
		self._storage = storage


	@classmethod
	def _make_tiles(self, shape, tilespec):
		"""
		>>> a = tiledArray((100,16,20), dtype=np.float32, tilespec=[[20,44,65,80],[5,10,15],[5,10,15,20]])
		>>> print(a._tiles[(1,1,1)])
		{'data': False, 'bounds': [(20, 44), (5, 10), (5, 10)]}
		"""

		tileshape = [len(bounds) -1 for bounds in tilespec]

		indices = np.array(np.ones(tileshape).nonzero()).T

		tiles =dict()

		for i in range(indices.shape[0]):

			indice = tuple(indices[i])

			bounds = []
			for d in range(len(shape)):
				bounds.append((tilespec[d][indice[d]], tilespec[d][indice[d] + 1]))

			tiles[indice] = {'bounds':bounds, 'data':False}

		return tiles


	def tiles(self, slices=None):
		"""
		Identify and return tile IDs intersecting a subset specified by slices
		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=[[5,10,15],[5,10,15]])
		>>> print(a._tilespec)
		[[0, 5, 10, 15, 16], [0, 5, 10, 15, 20]]

		>>> a.tiles((slice(0,4), slice(0,5)))
		{(0, 0): {'data': False, 'bounds': [(0, 5), (0, 5)]}}
		>>> a.tiles((slice(5,16), slice(0,6))).keys()
		[(3, 0), (3, 1), (2, 1), (2, 0), (1, 0), (1, 1)]
		>>> a.tiles((slice(15,16), slice(0,6))).keys()
		[(3, 0), (3, 1)]
		>>> a.tiles((slice(8,12), slice(16,23))).keys()
		[(1, 3), (2, 3)]
		>>> a.tiles((slice(8,-2), slice(16,23))).keys()
		[(1, 3), (2, 3)]
		>>> a.tiles((slice(8,-2), [2,8,19])).keys()
		[(1, 3), (2, 1), (2, 0), (2, 3), (1, 0), (1, 1)]
		"""

		slices = real_slices(self.shape, slices)

		result = dict()

		for index, tile in self._tiles.items():

			intersect = True

			for d in range(len(self.shape)):

				bounds = tile['bounds'][d]

				if isinstance(slices[d], slice):
					intersect = intersect and (slices[d].stop > bounds[0] and slices[d].start < bounds[1])

				else:
					intersect = intersect and np.array([(i >= bounds[0] and i < bounds[1]) for i in slices[d]]).any()

			if intersect:
				result[index] = tile

		return result


	def _tile_slices(self, tile, slices):
		"""Adjust slice values to tile relative slice"""

		tile_slices, data_slices = [], []

		slices = real_slices(self.shape, slices)

		for d in range(len(slices)):

			tile_start = max(slices[dim].start - tile['ranges'][dim,0], 0)
			tile_stop = min(slices[dim].stop - tile['ranges'][dim,0], tile['ranges'][dim,1] - tile['ranges'][dim,0])

			data_start = max(tile['ranges'][dim,0] - slices[dim].start, 0)
			data_stop = min(slices[dim].stop - slices[dim].start, tile['ranges'][dim,1] - slices[dim].start)

			tile_slices.append(slice(tile_start, tile_stop))
			data_slices.append(slice(data_start, data_stop))


		return tile_slices, data_slices


	def __setitem__(self, slices, values):
		""" Set values from the specified slices of the array
		>>> #a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> #a[0:16,0:20] = 42.0
		>>> #a[7,17]
		[[ 42.]]
		"""

		# Iterate through all tiles intersecting slices
		for index in self.find_tiles(slices):

			tile = self.tiles[index]

			tileshape = tuple(tile['ranges'][:,1] - tile['ranges'][:,0])
			tile_slices, data_slices = self._tile_slices(tile, slices)


			# Lazy allocation
			if not tile['data']:
				tile['data'] = self._storage((tileshape), dtype=self.dtype)
				tile['data'][:] = np.nan


			# For Arrays or np.ndarray assign using data_slices, otherwise assign as scalar value
			if isinstance(values, Array) or isinstance(values, np.ndarray):
				tile['data'][tile_slices] = values[data_slices]
			else:
				tile['data'][tile_slices] = values



	def __getitem__(self, slices):
		""" Get values from the specified slices of the array and return a new instance that
		references the original tiledArray.  Essentially returns a new view of the original

		>>> #a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> #a[7:14,3:12] = 42.0
		>>> #a[7,2:5]
		[[ nan  42.  42.]]
		"""

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(shape, self.dtype, tilespec=self._tilespec)
		result._view = view

		result.tiles = dict()

		for index in self.find_tiles(view):
			result.tiles[index] = self.tiles[index]

		return result


	def ndarray(self, astype=None):
		""" Returns a copy, default type is numpy ndarray of the array

		>>> #a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> #a[:] = np.arange(320).reshape((16,20))
		>>> #a[7,2:5]
		[[ 142.  143.  144.]]
		>>> #a[7,2:5] = 96
		>>> #a[6:7,0:5]
		[[ 120.  121.  122.  123.  124.]]
		>>> #a[6:7,0:5][:,1:3]
		[[ 121.  122.]]
		"""

		if astype != None:
			result = astype(self.shape, dtype=self.dtype, tilespec=tuple(outshape))
		else:
			result = np.empty(self.shape, dtype=self.dtype)

		for index in self.tiles:

			tile = self.tiles[index]

			tile_slices, data_slices = self._tile_slices(tile, self._view)

			if not tile['data']:
				data = np.nan

			else:
				data = tile['data'][tile_slices].ndarray()

			result[data_slices] = data

		return result

	def __repr__(self):
		return str(self.ndarray())

	def apply(self, func, axis=None):
		"""Apply an array function along successive axes
		>>> #a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> #a[:] = 42
		>>> #b = a.apply('mean', axis=0)
		>>> #b.shape
		(20,)
		>>> #b[3:7]
		[ 42.  42.  42.  42.]
		"""

		outshape = []
		tmpshape = []
		tilespec = []

		# Construct outshape and new tilespec by dropping axes
		for i in range(len(self.shape)):
			if i != axis:
				outshape.append(self.shape[i])
				tmpshape.append(self.shape[i])
				tilespec.append(self._tilespec[i])
			else:
				tmpshape.append(int(np.ceil(float(self.shape[i])/self._tilespec[i])))

		outshape = tuple(outshape)
		result = self.__class__(outshape, dtype=self.dtype, tilespec=tuple(tilespec))

		tmpshape = tuple(tmpshape)
		tmp = np.empty(tmpshape, dtype=self.dtype)

		result[:] = np.nan

		for index in self.find_tiles(self._view):
			tile = self.tiles[index]

			tile_slices, data_slices = self._tile_slices(tile, self._view)

			if axis != None:
				data_slices[axis] = index[axis]

			if not tile['data']:
				tmp[data_slices] = np.nan

			else:
				tmp[data_slices] = tile['data'][tile_slices].apply(func, axis=axis).ndarray()

		if func == 'mean':
			result[:] = tmp.mean(axis=axis)

		return result











