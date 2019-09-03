
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


ORDER = 1

# perform auto georeferencing
mapfit.automap('maps/test.png',
               source='natearth',
               textcolor=(0,0,0),
               warp_order=ORDER,
               )
gcps = pg.VectorData('maps/test_controlpoints.geojson')
frompoints = [(f['origx'],f['origy']) for f in gcps]
topoints = [(f['matchx'],f['matchy']) for f in gcps]
tiepoints = zip(frompoints, topoints)

# OR use the original tiepoints
##im = Image.open('maps/test.png')
##places = pg.VectorData('maps/test_placenames.geojson')
##tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
##warped = mapfit.main.warp(im, 'maps/test_georeferenced.tif', tiepoints, order=ORDER)
##cps = [('',oc,'',mc,[]) for oc,mc in tiepoints]
##mapfit.main.debug_warped('maps/test_georeferenced.tif', 'maps/test_debug_warp.png', cps)





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
# NOTE: shouldnt subsample the error raster, only temporary for visualization
xscale,xskew,xoff,yskew,yscale,yoff = actual.affine
xscale *= subsamp
yscale *= subsamp
error = pg.RasterData(width=actual.width/subsamp, height=actual.height/subsamp, mode='float32', 
                      affine=[xscale,xskew,xoff,yskew,yscale,yoff])
##error = pg.RasterData(width=actual.width, height=actual.height, mode='float32', 
##                      affine=actual.affine)
errband = error.add_band(nodataval=-99)
errband.compute('-99')

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

# estimating via transform
print 'estimating transform...'
coeff_x, coeff_y = mapfit.rmse.polynomial(ORDER, *zip(*tiepoints))[-2:]

print 'defining error grid...'
points = []
for row in range(error.height):
    #print (row, actual.height)
    for col in range(error.width):
        x,y = error.cell_to_geo(col,row)
        points.append((x,y))

print 'backwards resampling...'
points = np.array(points)
#print points
#print coeff_x, coeff_y
pred = mapfit.rmse.predict(ORDER, points, coeff_x, coeff_y, invert=True)

# experiment with inverting
##print 'pixels',pixels
##A = np.row_stack((coeff_x, coeff_y))
##for i in range(2, len(coeff_x)):
##    row = np.zeros(coeff_x.shape)
##    row[i] = 1
##    A = np.row_stack((A, row))
##print 'A',A
##print 'pred',pred
##inv = np.linalg.inv(A)
##print 'Ainv',inv
##coeff_x = inv[0,:]
##coeff_y = inv[1,:]
##for c in pred:
##    c = np.append(c, [0])
##    print c
##    recr = inv.dot(c)
##    print 'pred inv',recr
####recr = mapfit.rmse.predict(ORDER, pred, coeff_x, coeff_y)
####print 'pred inv',recr
##fsdafas
    
pred = pred.reshape((error.height, error.width,2))
#print pred.shape
#print pred

print 'calculating error distances between original and georeferenced'
arrows = pg.VectorData()
arrows.fields = ['dist']
from math import hypot
for row in range(error.height):
    #print (row, error.height)
    for col in range(error.width):
        if 1:
            x,y = error.cell_to_geo(col, row)
            origcol,origrow = pred[row,col]
            ix,iy = ideal.cell_to_geo(origcol, origrow)
            #dist = hypot(ix-x, iy-y)
            res = geod.Inverse(iy,ix,y,x)
            dist = res['s12']
            arrows.add_feature([dist], {'type':'LineString','coordinates':[(x,y),(ix,iy)]})
            #print (col,row), (iy,ix,y,x), dist, hypot(ix-x, iy-y)
            errband.set(col, row, dist)
        #except:
        #    pass

# render actual georefs over each other, with distortion arrows
m = pg.renderer.Map(2000, 2000)
m.add_layer('maps/test_georeferenced.tif')
m.add_layer('maps/test.tif', transparency=0.5)
m.add_layer(arrows, fillcolor='black', fillsize='1px')
m.zoom_auto()
m.add_layer(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor=None, outlinecolor='red')
#m.view()
m.save('maps/test_debug_warp2.png')

# render error surface map, with distortion arrows
stats = errband.summarystats()
print stats

m = error.render(2000, 2000)
#m.add_layer(r"C:\Users\kimok\Desktop\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor=None)
#m.add_layer(actual)
#m.zoom_bbox(*error.bbox)
m.add_layer(arrows, fillcolor='black', fillsize='1px') #, text=lambda f: f['dist'], textoptions={'textsize':5})
#m.view()
#fdsfds
m.add_legend()
m.save('maps/test_errgrid.tif')
