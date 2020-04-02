
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





print(os.getcwd())
try:
    os.chdir('simulations')
except:
    pass






##################
# FUNCTIONS

# Error output metrics
def error_output(georef_fil, truth_fil, geographic_error_surface, pixel_error_surface):
    print('Outputting error metrics')
    georef_root,ext = os.path.splitext(georef_fil)
    dct = {}


    ### known error surface avg + stdev
    dct['rmse_georeferenced'] = {}
    dct['max_georeferenced'] = {}
    # first geographic
    resids = np.array(geographic_error_surface.bands[0].img).flatten()
    valid = resids != geographic_error_surface.bands[0].nodataval
    resids = resids[valid]
    dct['rmse_georeferenced']['geographic'] = mapfit.accuracy.RMSE(resids)
    dct['max_georeferenced']['geographic'] = float(resids.max())
    # then pixels
    resids = np.array(pixel_error_surface.bands[0].img).flatten()
    valid = resids != pixel_error_surface.bands[0].nodataval
    resids = resids[valid]
    dct['rmse_georeferenced']['pixels'] = mapfit.accuracy.RMSE(resids)
    dct['max_georeferenced']['pixels'] = float(resids.max())
    # then percent (of image pixel dims)
    im = Image.open('maps/{}'.format(truth_fil))
    diag = hypot(*im.size)
    pixperc = dct['rmse_georeferenced']['pixels'] / float(diag)
    maxperc = dct['max_georeferenced']['pixels'] / float(diag)
    dct['rmse_georeferenced']['percent'] = pixperc
    dct['max_georeferenced']['percent'] = maxperc
    # ...
    

    ### controlpoint rmse
    with open('output/{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
    # first geog and pixels
    dct['rmse_controlpoints'] = {'geographic': transdict['forward']['error'],
                                 'pixels': transdict['backward']['error'],}
    # then percent (of image pixel dims)
    im = Image.open('maps/{}'.format(truth_fil))
    diag = hypot(*im.size)
    pixperc = dct['rmse_controlpoints']['pixels'] / float(diag)
    dct['rmse_controlpoints']['percent'] = pixperc

    
    ### (diff from orig controlpoints?)
    # ...


    ### all labels, for reference
    root = '_'.join(georef_root.split('_')[:4])
    allnames = pg.VectorData('maps/{}_placenames.geojson'.format(root))
    dct['labels'] = len(allnames)


    ### distribution of all labels? 
    # ... 


    ### percent of labels detected
    detected = pg.VectorData('output/{}_debug_text_toponyms.geojson'.format(georef_root))
    perc = len(detected) / float(len(allnames))
    dct['labels_detected'] = perc


    ### percent of labels used
    gcps = pg.VectorData('output/{}_controlpoints.geojson'.format(georef_root))
    perc = len(gcps) / float(len(allnames))
    dct['labels_used'] = perc


    ### distribution of controlpoints used? 
    # ... 
    

    return dct



# Main routine

def run_error_assessment(georef_fil, truth_fil):
    georef_root,ext = os.path.splitext(georef_fil)

    # original/simulated map
    truth = pg.RasterData('maps/{}'.format(truth_fil))
    print truth.affine
    
    # georeferenced/transformed map
    georef = pg.RasterData('output/{}'.format(georef_fil))
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    print georef.affine

    # calc error surface
    with open('output/{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
        sampling_trans = mapfit.transforms.from_json(transdict['backward']['model'])
    georef_errorsurf_geo = mapfit.debug.error_surface_georef(sampling_trans, georef, truth, 'geographic')
    georef_errorsurf_pix = mapfit.debug.error_surface_georef(sampling_trans, georef, truth, 'pixel')

    # output metrics
    errdct = error_output(georef_fil, truth_fil, georef_errorsurf_geo, georef_errorsurf_pix)
    with open('output/{}_error.json'.format(georef_root), 'w') as fobj:
        fobj.write(json.dumps(errdct))

    # visualize geographic error
    #mapp = error_vis(georef, georef_errorsurf_geo)
    #mapp.save('maps/{}_error_vis_geo.png'.format(georef_root))

    # visualize pixel error
    #mapp = error_vis(georef, georef_errorsurf_pix)
    #mapp.save('maps/{}_error_vis_pix.png'.format(georef_root))


def run_in_process(georef_fil, truth_fil):
    georef_root,ext = os.path.splitext(georef_fil)
    logger = codecs.open('output/{}_error_log.txt'.format(georef_root), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))
    
    run_error_assessment(georef_fil, truth_fil)


def itermaps():
    for fil in os.listdir('maps'):
        if '_image.' in fil and fil.endswith(('.png','jpg')):
            yield fil



####################
# RUN

if __name__ == '__main__':

    maxprocs = 4
    procs = []

    for imfil in itermaps():
        fil_root = imfil.split('_image.')[0]


        # LOCAL 

        ## auto
        autofil = '{}_georeferenced_auto.tif'.format(fil_root)
        print(imfil,autofil)
        if os.path.lexists('output/{}'.format(autofil)):
            run_error_assessment(autofil,imfil)

        ## exact
##        exactfil = '{}_georeferenced_exact.tif'.format(fil_root)
##        if os.path.lexists('maps/{}'.format(exactfil)):
##            gcps = '{}_georeferenced_exact_controlpoints.geojson'.format(fil_root)
##            run_error_assessment(exactfil,imfil,gcps)

        continue
            

        # MULTIPROCESS

        ## auto
        autofil = '{}_georeferenced_auto.tif'.format(fil_root)
        print(imfil,autofil)
        if os.path.lexists('output/{}'.format(autofil)):
            p = mp.Process(target=run_in_process,
                           args=(autofil,imfil),
                           )
            p.start()
            procs.append(p)

        ## exact
##        exactfil = '{}_georeferenced_exact.tif'.format(fil_root)
##        if os.path.lexists('maps/{}'.format(exactfil)):
##            gcps = '{}_georeferenced_exact_controlpoints.geojson'.format(fil_root)
##            p = mp.Process(target=run_in_process,
##                           args=(exactfil,imfil,gcps),
##                           )
##            p.start()
##            procs.append(p)

        # Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)

        



