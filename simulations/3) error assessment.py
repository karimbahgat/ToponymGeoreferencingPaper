
import automap as mapfit
import pythongis as pg
from PIL import Image
import numpy as np
from geographiclib.geodesic import Geodesic
from math import hypot
import codecs

import multiprocessing as mp
import sys
import os
import datetime
import json

# Measure positional error surface bw original simulated map coordinates and georeferenced map coordinates

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





##################
# FUNCTIONS

# Error surface calculation
# EITHER dataset centric, since the purpose of georef is data capture, to see if we can recreate the original data
# ...measured as avg deviation bw coordinates, one metric per dataset layer
# ...
# OR map centric, generate an error grid that captures difference bw truth and georef transformed pixel.
# ...should be same as dataset centric, but also allows for spatial error variation.
# ...during simulation, truth pixel can be calculated by storing drawing affine along with rendered image, and querying each pixel coordinate.
# ...or without the simulation, truth pixel can be proxied by...
def error_surface(i, truth, georef):    
    print('Creating error surface')
    # create coordinate distortion grid as a smaller sampling of original grid, since dist calculations are slow
    # NOTE: shouldnt subsample the error raster, only temporary for visualization
    xscale,xskew,xoff,yskew,yscale,yoff = georef.affine
    xscale *= subsamp
    yscale *= subsamp
    error = pg.RasterData(width=georef.width/subsamp, height=georef.height/subsamp, mode='float32', 
                          affine=[xscale,xskew,xoff,yskew,yscale,yoff])
    errband = error.add_band(nodataval=-99)
    errband.compute('-99')

    # get tiepoints from automated tool
    gcps = pg.VectorData('maps/sim_{}_image_controlpoints.geojson'.format(i))
    frompoints = [(f['origx'],f['origy']) for f in gcps]
    topoints = [(f['matchx'],f['matchy']) for f in gcps]
    tiepoints = zip(frompoints, topoints)
    # OR use the georef coordinates for the rendered placenames (should be approx 0 error...)
    ##places = pg.VectorData('maps/test_placenames.geojson')
    ##tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]

    # estimating map transform
    print('estimating transform from tiepoints...')
    coeff_x, coeff_y = mapfit.rmse.polynomial(ORDER, *zip(*tiepoints))[-2:]

    #
    print('defining error sampling points...')
    points = []
    for row in range(error.height):
        #print (row, georef.height)
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
                ix,iy = truth.cell_to_geo(origcol, origrow)
                #dist = hypot(ix-x, iy-y)
                res = geod.Inverse(iy,ix,y,x)
                dist = res['s12']
                arrows.add_feature([dist], {'type':'LineString','coordinates':[(x,y),(ix,iy)]})
                #print (col,row), (iy,ix,y,x), dist, hypot(ix-x, iy-y)
                errband.set(col, row, dist)
            #except:
            #    pass

    return error, arrows

# Error output metrics
def error_output(i, error):
    print('Outputting error metrics')
    band = error.bands[0]
    avg = band.summarystats('mean')['mean']
    devs = (abs(avg - cell.value) for cell in band)
    cnt = band.width * band.height
    stdev = sum(devs) / float(cnt)

    dct = {'avg':avg, 'stdev':stdev}
    with open('maps/sim_{}_error.json'.format(i), 'w') as fobj:
        fobj.write(json.dumps(dct))

# Visualize differences
def error_vis(i, error, arrows, georef):
    print('Visualizing original vs georeferenced errors')

    # render georef georefs over each other, with distortion arrows
##    print('overlay images with distortion arrows')
##    m = pg.renderer.Map(2000, 2000)
##    m.add_layer('maps/test_georeferenced.tif')
##    m.add_layer('maps/test.tif', transparency=0.5)
##    m.add_layer(arrows, fillcolor='black', fillsize='1px')
##    m.zoom_auto()
##    m.add_layer(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor=None, outlinecolor='red')
##    m.save('maps/test_debug_warp2.png')

    # render error surface map, with distortion arrows?
    print('error color surface and values')
    m = pg.renderer.Map(background='white')
    m.add_layer(georef)
    m.add_layer(error, transparency=0.2, legendoptions={'title':'Error (Meters)', 'valueformat':'.0f'})
    #m.add_layer(arrows, fillcolor='black', fillsize='1px', legend=False) #, text=lambda f: f['dist'], textoptions={'textsize':5})
    m.zoom_bbox(*georef.bbox)
    m.add_legend({'padding':0})
    m.save('maps/sim_{}_error_vis.png'.format(i))

def error_assess(i):
    logger = codecs.open('maps/sim_{}_error_log.txt'.format(i), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))

    # original/simulated map
    truth = pg.RasterData('maps/sim_{}_truth.tif'.format(i))
    
    # georeferenced/transformed map
    georef = pg.RasterData('maps/sim_{}_image_georeferenced.tif'.format(i))

    print truth.affine
    print georef.affine

    # surface
    error,arrows = error_surface(i, truth, georef)
    #error.save('maps/sim_{}_error.tif'.format(i))

    # output metrics
    error_output(i, error)

    # visualize
    error_vis(i, error, arrows, georef)




####################
# RUN

if __name__ == '__main__':

    maxprocs = 8
    procs = []

    for i in range(1, 250):
        print(i)

        #error_assess(i)
        #continue

        # Begin process
        p = mp.Process(target=error_assess,
                       args=(i,),
                       )
        p.start()
        procs.append(p)

        # Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)

        



