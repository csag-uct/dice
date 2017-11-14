import numpy as np

from array import Array
from array import real_slices, reslice

from nparray import numpyArray





class tiledArray(Array):
	"""
	An Array implemention that uses numpy arrays to store subsets/tiles of the full data
	an supports sparse tile sets (some tiles can be missing) and parallel processing
	across tiles

	tiles are managed through a dict
	"""

	def __init__(self, shape, dtype, tilespec=None, storage=numpyArray):
		"""
		Create an instance of a tiled Array.  Sets up the tile dictionary and tile bounds arrays
		but doesn't allocate and storage as allocation is lazy

		>>> a = tiledArray((100,16,20), dtype=np.float32, tilespec=(20,5,5))
		"""

		super(tiledArray, self).__init__(shape, dtype)

		if tilespec:
			self._tilespec = tilespec
		else:
			self._tilespec = shape

		self.tiles = self._make_tiles(self._tilespec)
		self._storage = storage

	def _make_tiles(self, tilespec):

		shape = tuple([int(np.ceil(float(self.shape[i])/tilespec[i])) for i in range(len(self.shape))])
		indices = np.array(np.ones(shape).nonzero()).T
		tiles = dict()

		for i in range(0, indices.shape[0]):
			index = tuple(indices[i])

			ranges = np.zeros((len(self.shape),2), dtype=np.int)

			for d in range(len(self.shape)):
				start = index[d] * tilespec[d]
				stop = (index[d] + 1) * tilespec[d]
				stop = min(stop, self.shape[d])
				ranges[d] = np.array([start, stop], dtype=np.int)

			tiles[index] = {'ranges':ranges, 'data':False}

		return tiles


	def find_tiles(self, slices):
		"""
		Identify and return tile IDs intersecting a subset specified by slices
		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> a.find_tiles((slice(0,16), slice(0,5)))
		[(0, 0), (3, 0), (2, 0), (1, 0)]
		"""

		slices = real_slices(self.shape, slices)

		starts = np.array([s.start for s in slices])
		stops = np.array([s.stop for s in slices])

		result = []
		for index, tile in self.tiles.items():

			ranges = tile['ranges']

			if np.all(np.logical_and(ranges[:,1] > starts, ranges[:,0] < stops)):
				result.append(index)

		return result



	def _tile_slices(self, tile, slices):
		"""Adjust slice values to tile relative slice"""

		tile_slices, data_slices = [], []

		slices = real_slices(self.shape, slices)

		for dim in range(len(slices)):
			
			tile_start = max(slices[dim].start - tile['ranges'][dim,0], 0)
			tile_stop = min(slices[dim].stop - tile['ranges'][dim,0], tile['ranges'][dim,1] - tile['ranges'][dim,0])

			data_start = max(tile['ranges'][dim,0] - slices[dim].start, 0)
			data_stop = min(slices[dim].stop - slices[dim].start, tile['ranges'][dim,1] - slices[dim].start)

			tile_slices.append(slice(tile_start, tile_stop))
			data_slices.append(slice(data_start, data_stop))


		return tile_slices, data_slices


	def __setitem__(self, slices, values):
		""" Set values from the specified slices of the array
		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> a[0:16,0:20] = 42.0
		>>> a[7,17]
		tiledArray([[ 42.]])
		"""
#		print self.tiles, self._view, slices, values
		
		# Iterate through all tiles intersecting slices
		for index in self.find_tiles(slices):

			tile = self.tiles[index]

			tileshape = tuple(tile['ranges'][:,1] - tile['ranges'][:,0])
			tile_slices, data_slices = self._tile_slices(tile, slices)

#			print index, tileshape, tile_slices, data_slices

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

		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> a[7:14,3:12] = 42.0
		>>> a[7,2:5]
		tiledArray([[ nan  42.  42.]])
		"""

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(shape, self.dtype, tilespec=self._tilespec)
		result._view = view

		result.tiles = dict()
		
		for index in self.find_tiles(view):
			result.tiles[index] = self.tiles[index]

		return result


	def copy(self, astype=None):
		""" Returns a copy, default type is numpy ndarray of the array
		
		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> a[:] = np.arange(320).reshape((16,20))
		>>> a[7,2:5]
		tiledArray([[ 142.  143.  144.]])
		>>> a[7,2:5] = 96
		>>> a[6:7,0:5]
		tiledArray([[ 120.  121.  122.  123.  124.]])
		>>> a[6:7,0:5][:,1:3]
		tiledArray([[ 121.  122.]])
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
				data = tile['data'][tile_slices].copy()

			result[data_slices] = data

		return result

	def __repr__(self):
		return "tiledArray({})".format(str(self.copy()))

	def apply(self, func, axis=None):
		"""Apply an array function along successive axes
		>>> a = tiledArray((16,20), dtype=np.float32, tilespec=(5,5))
		>>> a[:] = 42
		>>> b = a.apply('mean', axis=0)
		>>> b.shape
		(20,)
		>>> b[3:7]
		tiledArray([ 42.  42.  42.  42.])
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
				tmp[data_slices] = tile['data'][tile_slices].apply(func, axis=axis).copy()

		if func == 'mean':
			result[:] = tmp.mean(axis=axis)

		return result












