import numpy as np

from ddice.array import Array
from ddice.array import reslice
from ddice.array import tiledArray

from ddice.variable import Variable

import psycopg2
from psycopg2.pool import ThreadedConnectionPool


class pgArray(Array):
	"""
	An array class that uses postgresql databaseas array storage

	>>> A = pgArray((500,300), dtype=np.float32, dsn='postgresql://postgres:Dweeb&2684@localhost/dice-data')
	>>> A.shape
	(500, 300)
	>>> A[:] = np.arange(500*300).reshape((500,300))
	>>> print A[250,150]
	[[ 75150.]]

	>>> A[250,150] = 42.0
	>>> print A[250,150]
	[[ 42.]]

	>>> B = tiledArray((100,160,200), dtype=np.float32, tilespec=(20,25,25), storage=pgArray)
	>>> B[:] = np.arange(100*160*200, dtype=np.float32).reshape((100,160,200))
	>>> print B[50,80,100]
	[[[ 1616100.]]]

	"""

	pools = {}

	@classmethod
	def get_connection(cls, dsn):

		if dsn in cls.pools:
			return cls.pools[dsn].getconn()

		else:
			cls.pools[dsn] = ThreadedConnectionPool(10,100,dsn)
			return cls.pools[dsn].getconn()

	@classmethod
	def put_connection(cls, dsn, conn):
		cls.pools[dsn].putconn(conn)

	def __init__(self, shape, dtype, dsn='postgresql://postgres:Dweeb&2684@localhost/dice-data'):

		super(pgArray, self).__init__(shape, dtype)

		self._dsn = dsn

		# Create a connection pool
#		self._conn = psycopg2.connect(dsn)

		# We'll lazy allocate
		self.__data = False


	def __setitem__(self, slices, values):
		""" Set values from the specified slices of the array
		"""

		# Get db cursor
		conn = self.__class__.get_connection(self._dsn)
		cur = conn.cursor()
		
		# Combine slices with view to get data indices
		shape, slices = reslice(self.shape, self._view, slices)

		# Lazy allocation means we need to create an empty array first
		if type(self.__data) == bool and self.__data == False:

			data = np.empty(self.shape, dtype=self.dtype)

			data[slices] = values
			
			cur.execute('insert into arrays (shape, dtype, data) values (%s, %s, %s) returning id;', (list(self.shape), 0, psycopg2.Binary(data)))
			self.__data = cur.fetchone()[0]

		# If not then we first retreive data blob, update it, then update back to the db (this is not very optimal!)
		else:
			cur.execute('select * from arrays where id = %s;', (self.__data,))
			result = cur.fetchone()

			data = np.frombuffer(result[3], dtype=np.float32).reshape(result[1]).copy()

			data[slices] = values

			cur.execute('update arrays set data = %s where id = %s;', (psycopg2.Binary(data), self.__data))

		conn.commit()
		self.__class__.put_connection(self._dsn, conn)


	def __getitem__(self, slices):

		# Calculate new view 
		shape, view = reslice(self.shape, self._view, slices)

		# Create new array instance
		result = self.__class__(shape, self.dtype, self._dsn)
		
		# Set new array instance view to the new view
		result._view = view
		result.__data = self.__data

		return result

	def __repr__(self):
		return str(self.ndarray())

	@property
	def _data(self):

		# Get db cursor
		conn = self.__class__.get_connection(self._dsn)
		cur = conn.cursor()
		
		if type(self.__data) == bool and self.__data == False:

			data = np.empty(self.shape, dtype=self.dtype)

			cur.execute('insert into arrays (shape, dtype, data) values (%s, %s, %s) returning id;', (list(self.shape), 0, psycopg2.Binary(data)))
			self.__data = cur.fetchone()[0]

		else:
			cur.execute('select * from arrays where id = %s;', (self.__data,))
			result = cur.fetchone()

			data = np.frombuffer(result[3], dtype=np.float32).reshape(result[1])

		conn.commit()
		self.__class__.put_connection(self._dsn, conn)


		return data



class pgVariable(Variable, object):
	"""
	A netCDF4 file based variable
	"""

	def __init__(self, dimensions, dtype, name=None, attributes={}, dataset=None, data=None):
		"""
		Just call the parent class constructor but force storage to be netCDF4Array
		"""
		
		super(pgVariable, self).__init__(dimensions, dtype, name=name, attributes=attributes, dataset=dataset, data=data, storage=pgArray)



class pgDataset(Dataset, object):
	"""
	A postgresql Dataset implementation
	"""

	pools = {}

	@classmethod
	def get_connection(cls, dsn):

		if dsn in cls.pools:
			return cls.pools[dsn].getconn()

		else:
			cls.pools[dsn] = ThreadedConnectionPool(10,100,dsn)
			return cls.pools[dsn].getconn()

	@classmethod
	def put_connection(cls, dsn, conn):
		cls.pools[dsn].putconn(conn)


	def __init__(self, dsn, dataset=None):

		self._dimensions = []
		self._attributes = {}
		self._variables = {}


		# If there is no Dataset instance provided then assume we are opening an existing dataset
		if not dataset:

			self._dsn = dsn
			


