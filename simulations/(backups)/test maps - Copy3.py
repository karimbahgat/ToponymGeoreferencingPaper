
import automap as mapfit
import pythongis as pg
from PIL import Image
import numpy as np

# OUTLINE
# vary the following:
# - projection difference
# - gazetteer source coord differences
# - warp method/order

# for instance:
# - no projection difference + same gazetteer (should eliminate error from placename matching?)
#   (by holding these constant, any error should be from technical details, like rounding, point detection, warp bugs???)
#   - comparing with self, test should return 0 error
#   - prespecified georef, just use known placename coords as input controlpoints?? any error should be from the warp process?? 
#   - auto georeferencing (total error from auto approach, probably mostly from placename matching?? or not if using same gazetteer??)



# perform auto georeferencing
##mapfit.automap('maps/test.png',
##               source='natearth',
##               textcolor=(0,0,0),
##               warp_order=1,
##               )
##gcps = pg.VectorData('maps/test_controlpoints.geojson')
##frompoints = [(f['origx'],f['origy']) for f in gcps]
##topoints = [(f['matchx'],f['matchy']) for f in gcps]
##tiepoints = zip(frompoints, topoints)

# OR manual tiepoint georef
im = Image.open('maps/test.png')
places = pg.VectorData('maps/test_placenames.geojson')
tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
mapfit.main.warp(im, 'maps/test_georeferenced.tif', tiepoints, order=1)

# true georeferencing error is difference bw original and transformed data coordinates

####################
# 1) dataset centric, since the purpose of georef is data capture, to see if we can recreate the original data
# ...measured as avg deviation bw coordinates, one metric per dataset layer


####################
# 2) or map centric, generate an error grid that captures difference bw ideal and actual transformed pixel.
# ...should be same as dataset centric, but also allows for spatial error variation.
# ...during simulation, ideal pixel can be calculated by storing drawing affine along with rendered image, and querying each pixel coordinate.
# ...or without the simulation, ideal pixel can be proxied by...

from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

ideal = pg.RasterData('maps/test.tif')
actual = pg.RasterData('maps/test_georeferenced.tif')
print ideal.affine
print actual.affine

subsamp = 10 # create error grid as a smaller sampling of original grid, since dist calculations are slow
xscale,xskew,xoff,yskew,yscale,yoff = actual.affine
xscale *= subsamp
yscale *= subsamp
error = pg.RasterData(width=actual.width/subsamp, height=actual.height/subsamp, mode='float32',
                      affine=[xscale,xskew,xoff,yskew,yscale,yoff])
errband = error.add_band()

##from math import hypot
##for row in range(0, actual.height, subsamp):
##    print (row,ideal.height)
##    for col in range(0, actual.width, subsamp):
##        try:
##            x,y = actual.cell_to_geo(col, row)
##            icol,irow = ideal.geo_to_cell(x, y)
##            #dist = hypot(ix-x, iy-y)
##            res = geod.Inverse(iy,ix,y,x)
##            dist = res['s12']
##            #print (iy,ix,y,x),dist
##            errband.set(col/subsamp, row/subsamp, dist)
##        except:
##            pass

print 'estimating transform...'
coeff_x, coeff_y = mapfit.rmse.polynomial(1, *zip(*tiepoints))[-2:]

print 'sampling...'
pixels = []
for row in range(0, error.height):
    print (row, error.height)
    for col in range(0, error.width):
        x,y = error.cell_to_geo(col, row)
        icol,irow = ideal.geo_to_cell(x, y)
        pixels.append((icol,irow))

print 'predicting...'
pixels = np.array(pixels)
pred = mapfit.rmse.predict(1, pixels, coeff_x, coeff_y)
pred = pred.reshape((error.height,error.width,2))
print pred.shape

for row in range(0, error.height):
    print (row, error.height)
    for col in range(0, error.width):
        x,y = error.cell_to_geo(col, row)
        try:
            ix,iy = pred[row,col]
            res = geod.Inverse(iy,ix,y,x)
            dist = res['s12']
            #print (col,row),(iy,ix,y,x),dist
            errband.set(col, row, dist)
        except:
            pass
        
stats = errband.summarystats()
print stats

m = error.render(2000, 2000)
m.add_legend()
m.save('maps/test_errgrid.tif')
