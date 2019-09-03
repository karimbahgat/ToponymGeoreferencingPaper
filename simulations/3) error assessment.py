
import automap as mapfit
import pythongis as pg
from PIL import Image
import numpy as np
from geographiclib.geodesic import Geodesic
from math import hypot

# OUTLINE
# for instance:
# - no projection difference + same gazetteer (should eliminate error from placename matching?)
#   (by holding these constant, any error should be from technical details, like rounding, point detection, warp bugs???)
#   - comparing with self, test should return 0 error
#   - prespecified georef, just use known placename coords as input controlpoints?? any error should be from the warp process?? 
#   - auto georeferencing (total error from auto approach, probably mostly from placename matching?? or not if using same gazetteer??)



###################
# PARAMS
ORDER = 1
subsamp = 10

####################
# 1: Render some map with known parameters
# see "generate maps.py"




####################
# 2: Georeference the map
print('2: Georeferencing map')

# EITHER automated tool
mapfit.automap('maps/test.png',
               source='natearth',
               textcolor=(0,0,0),
               warp_order=ORDER,
               )
gcps = pg.VectorData('maps/test_controlpoints.geojson')
frompoints = [(f['origx'],f['origy']) for f in gcps]
topoints = [(f['matchx'],f['matchy']) for f in gcps]
tiepoints = zip(frompoints, topoints)

# OR use the actual coordinates for the rendered placenames (should be approx 0 error...)
##im = Image.open('maps/test.png')
##places = pg.VectorData('maps/test_placenames.geojson')
##tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
##warped = mapfit.main.warp(im, 'maps/test_georeferenced.tif', tiepoints, order=ORDER)
##cps = [('',oc,'',mc,[]) for oc,mc in tiepoints]
##mapfit.main.debug_warped('maps/test_georeferenced.tif', 'maps/test_debug_warp.png', cps)





##################
# 3: Measure positional error surface bw original simulated map coordinates and georeferenced map coordinates
print('3: Measuring positional error')

# EITHER dataset centric, since the purpose of georef is data capture, to see if we can recreate the original data
# ...measured as avg deviation bw coordinates, one metric per dataset layer
# ...

# OR map centric, generate an error grid that captures difference bw ideal and actual transformed pixel.
# ...should be same as dataset centric, but also allows for spatial error variation.
# ...during simulation, ideal pixel can be calculated by storing drawing affine along with rendered image, and querying each pixel coordinate.
# ...or without the simulation, ideal pixel can be proxied by...

print('setting up...')
# original/simulated map
ideal = pg.RasterData('maps/test.tif')
# georeferenced/transformed map
actual = pg.RasterData('maps/test_georeferenced.tif')
print ideal.affine
print actual.affine
# create coordinate distortion grid as a smaller sampling of original grid, since dist calculations are slow
# NOTE: shouldnt subsample the error raster, only temporary for visualization
xscale,xskew,xoff,yskew,yscale,yoff = actual.affine
xscale *= subsamp
yscale *= subsamp
error = pg.RasterData(width=actual.width/subsamp, height=actual.height/subsamp, mode='float32', 
                      affine=[xscale,xskew,xoff,yskew,yscale,yoff])
errband = error.add_band(nodataval=-99)
errband.compute('-99')

# estimating map transform
print('estimating transform from tiepoints...')
coeff_x, coeff_y = mapfit.rmse.polynomial(ORDER, *zip(*tiepoints))[-2:]

#
print('defining error sampling points...')
points = []
for row in range(error.height):
    #print (row, actual.height)
    for col in range(error.width):
        x,y = error.cell_to_geo(col,row)
        points.append((x,y))

#
print('inverting transform matrix for backwards resampling (georeferenced pixels to original image pixels)...')
points = np.array(points)
#print points
#print coeff_x, coeff_y
pred = mapfit.rmse.predict(ORDER, points, coeff_x, coeff_y, invert=True)
pred = pred.reshape((error.height, error.width,2))
#print pred.shape
#print pred

#
print('measuring error distances between original and georeferenced')
geod = Geodesic.WGS84
arrows = pg.VectorData()
arrows.fields = ['dist']
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
stats = errband.summarystats()
print stats





##################
# 4: Visualize differences
print('4: Visualizing original vs georeferenced errors')

# render actual georefs over each other, with distortion arrows
print('overlay images with distortion arrows')
m = pg.renderer.Map(2000, 2000)
m.add_layer('maps/test_georeferenced.tif')
m.add_layer('maps/test.tif', transparency=0.5)
m.add_layer(arrows, fillcolor='black', fillsize='1px')
m.zoom_auto()
m.add_layer(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor=None, outlinecolor='red')
m.save('maps/test_debug_warp2.png')

# render error surface map, with distortion arrows
print('error color surface and values')
m = error.render(2000, 2000)
m.add_layer(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor=None, legend=False)
m.add_layer(arrows, fillcolor='black', fillsize='1px', legend=False) #, text=lambda f: f['dist'], textoptions={'textsize':5})
m.zoom_bbox(*error.bbox)
m.add_legend({'padding':0})
m.save('maps/test_errgrid.tif')




