import sys
sys.path.append('/home/cjack/work/projects/code/dice/')

import json

from dice.dataset import netCDF4Dataset
from dice.field import CFField

from flask import Flask
from flask import request


DATA_ROOT = '../dice/testing/'

app = Flask(__name__)


@app.route('/dataset/<path:path>')
def dataset(path):

	uri = "{}/{}".format(DATA_ROOT, path)

	ds = netCDF4Dataset(uri)

	return repr(ds.asjson())


@app.route('/dataset/<path:path>/features')
def features(path):

	uri = "{}/{}".format(DATA_ROOT, path)

	ds = netCDF4Dataset(uri)
	variables = ds.variables

	field = CFField(variables[variables.keys()[0]])

	data = request.args.get('data', None)

	return json.dumps(field.features())

@app.route('/dataset/<path:path>/map')
def map(path):

	uri = "{}/{}".format(DATA_ROOT, path)


	return json.dumps(field.features(data=data))
