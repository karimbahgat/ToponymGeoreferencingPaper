
from automap.rmse import polynomial, predict

import pythongis as pg

import numpy as np

###

mapp = 'burkina'
#mapp = 'brazil_land_1977'
#mapp = 'brazil_pol_1981'
gcps = pg.VectorData('testmaps/{}_controlpoints.geojson'.format(mapp))
rast = pg.RasterData('testmaps/{}_georeferenced.tif'.format(mapp))

# estimate coeffs
order = 3
frompoints = [(f['origx'],f['origy']) for f in gcps]
topoints = [(f['matchx'],f['matchy']) for f in gcps]
coeff_x, coeff_y = polynomial(order, frompoints, topoints)[-2:]

# predict origs
points = np.array(frompoints)
pred = predict(order, points, coeff_x, coeff_y)
print topoints
print pred

# predict rast
points = np.array([(x,y) for y in range(rast.height) for x in range(rast.width)])
pred = predict(order, points, coeff_x, coeff_y)
print len(pred)

# rast coords
real = np.array([(cell.x,cell.y) for cell in rast.bands[0]])
xdiff = real[:,0]-pred[:,0]
ydiff = real[:,1]-pred[:,1]
dists = np.sqrt(xdiff**2 + ydiff**2)
print dists

# show
from PIL import Image, ImageOps
print dists.min(), dists.mean(), dists.max()
im = Image.fromarray((dists*(255.0/dists.max())).reshape((rast.height, rast.width)).T)
im.show()
