import sys
sys.path.append('/home/cjack/work/projects/code/dice/')

import json
import numpy as np

from ddice.dataset import netCDF4Dataset
from ddice.field import CFField

from flask import Flask
from flask import request
from flask import Response

with open('catalogue.json') as cat_file:

	catalogue = json.loads(cat_file.read())['datasets']

class numpyEncoder(json.JSONEncoder):

	def default(self, obj):
		if isinstance(obj, np.float32):
			return float(obj)
		if isinstance(obj, np.int32):
			return int(obj)
		if isinstance(obj, np.int64):
			return int(obj)
		return json.JSONEncoder.default(self, obj)


app = Flask(__name__)


@app.route('/dataset/')
def datasets():

	return repr(catalogue)


@app.route('/dataset/<shortname>')
def dataset(shortname):

	properties = catalogue[shortname]
	app.logger.info(properties['path'])
	ds = netCDF4Dataset(properties['path'])

	app.logger.info(ds.asjson())
	return Response(json.dumps(ds.asjson(), cls=numpyEncoder), mimetype='application/json')

@app.route('/dataset/<shortname>/<varname>')
def variable(shortname, varname):

	properties = catalogue[shortname]

	ds = netCDF4Dataset(properties['path'])
	variable = ds.variables[varname]

	return Response(json.dumps(variable.asjson(), cls=numpyEncoder), mimetype='application/json')

@app.route('/dataset/<shortname>/features')
def features(shortname):

	properties = catalogue[shortname]

	if 'geometry' in properties:
		shapefile = properties['geometry']

	ds = netCDF4Dataset(properties['path'])
	variable = ds.variables[properties['variables'][0]]

	field = CFField(variable)

	data = request.args.get('data', None)

	features = field.feature_collection(values=True, shapefile=shapefile)
	app.logger.info(features)

	return Response(json.dumps(features, cls=numpyEncoder), mimetype='application/json')

@app.route('/dataset/<path:path>/map')
def map(path):

	uri = "{}/{}".format(DATA_ROOT, path)


	return json.dumps(field.features(data=data))
