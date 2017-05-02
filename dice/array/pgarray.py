import numpy as np

from array import Array, real_slices, reslice
from tiledarray import tiledArray

import psycopg2


class pgArray(Array):
	"""
	An array class that uses postgresql databaseas array storage

	>>> A = pgArray((500,300), dtype=np.float32, dsn='postgresql://postgres:Dweeb&2684@localhost/dice-data')
	>>> A.shape
	(500, 300)
	>>> A[:] = np.arange(500*300).reshape((500,300))
	>>> A[250,150].copy()
	array([[ 75150.]], dtype=float32)
	>>> A[250,150] = 42.0
	>>> A[250,150]
	[[ 42.]]
	>>> B = tiledArray((100,16,20), dtype=np.float32, tilespec=(20,5,5), storage=pgArray)
	>>> B[:] = np.arange(100*16*20, dtype=np.float32).reshape((100,16,20))
	>>> B[0,0,1]
	"""

	def __init__(self, shape, dtype, dsn='postgresql://postgres:Dweeb&2684@localhost/dice-data'):

		super(pgArray, self).__init__(shape, dtype)

		self._dsn = dsn
		self._conn = psycopg2.connect(dsn)

		# We'll lazy allocate
		self._data = False


	def __setitem__(self, slices, values):
		""" Set values from the specified slices of the array
		"""

		cur = self._conn.cursor()
		shape, slices = reslice(self.shape, self._view, slices)

		if type(self._data) == bool:
			lazy = True
		else:
			lazy = False

		# Lazy allocation
		if lazy:			
			data = np.empty(self.shape, dtype=self.dtype)
			data[slices] = values
			
			cur.execute('insert into arrays (shape, dtype, data) values (%s, %s, %s) returning id;', (list(self.shape), 0, psycopg2.Binary(data)))
			self._data = cur.fetchone()[0]

		else:
			cur.execute('select * from arrays where id = %s;', (self._data,))
			result = cur.fetchone()
			data = np.frombuffer(result[3], dtype=np.float32).reshape(result[1]).copy()

			# Combine slices with the view to get data indices
			data[slices] = values

			cur.execute('update arrays set data = %s where id = %s;', (psycopg2.Binary(data), self._data))

		self._conn.commit()


	def __getitem__(self, slices):

		shape, view = reslice(self.shape, self._view, slices)

		result = self.__class__(shape, self.dtype, self._dsn)
		result._view = view
		result._data = self._data

		return result

	def __repr__(self):
		return "pgArray({})".format(str(self.copy()))

	def copy(self, astype=None):

		cur = self._conn.cursor()
		
		if type(self._data) == bool:
			lazy = True
		else:
			lazy = False

		# Lazy allocation
		if lazy:			
			
			data = np.empty(self.shape, dtype=self.dtype)

			cur.execute('insert into arrays (shape, dtype, data) values (%s, %s, %s) returning id;', (list(self.shape), 0, psycopg2.Binary(data)))
			self._data = cur.fetchone()[0]

		else:
			cur.execute('select * from arrays where id = %s;', (self._data,))
			result = cur.fetchone()

			data = np.frombuffer(result[3], dtype=np.float32).reshape(result[1])


		return data[self._view]
