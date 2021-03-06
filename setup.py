from setuptools import setup

setup(name='ddice',
      version='0.1',
      description='Multi-dimensional data slicing and dicing, data management and serving',
      url='https://github.com/csag-uct/ddice',
      author='Christopher Jack',
      author_email='cjack@csag.uct.ac.za',
      license='MIT',
      packages=['ddice'],
      install_requires=[
          'netCDF4',
          'cfunits',
          'numpy',
          'shapely',
          'fiona',
          'pyproj'
      ],
      zip_safe=False)
