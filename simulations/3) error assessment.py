
import automap as mapfit
import pythongis as pg
from PIL import Image
import numpy as np
from math import hypot
import codecs

import multiprocessing as mp
import sys
import os
import datetime
import json
import gc

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





#################
# PARAMETERS

MAXPROCS = 20 # number of available cpu cores / parallel processes






##################
# FUNCTIONS

# Error output metrics
def scat_metric(points):
    # based on the scat measure from goncalves2009measures
    # the avg of median of each point's distance to all others should be close to half the avg of the image dimensions
    meds = []
    for p in points:
        dists = []
        for p2 in points:
            if p is p2: continue
            dx = p[0] - p2[0]
            dy = p[1] - p2[1]
            d = hypot(dx, dy)
            dists.append(d)
        med = sorted(dists)[len(dists)//2] # (lazy) median dist
        meds.append(med)
    scat = sum(meds)/float(len(meds)) # avg of medians
    return scat

def error_output(georef_fil, truth_fil, geographic_error_surface, pixel_error_surface):
    print('Outputting error metrics')
    georef_root,ext = os.path.splitext(georef_fil)
    dct = {}


    ### known error surface avg + stdev
    # first geographic
    dct['geographic'] = {}
    resids = np.array(geographic_error_surface.bands[0].img).flatten()
    valid = resids != geographic_error_surface.bands[0].nodataval
    resids = resids[valid]
    dct['geographic']['rmse'] = mapfit.accuracy.RMSE(resids)
    dct['geographic']['max'] = float(resids.max())
    # then pixels
    dct['pixel'] = {}
    resids = np.array(pixel_error_surface.bands[0].img).flatten()
    valid = resids != pixel_error_surface.bands[0].nodataval
    resids = resids[valid]
    dct['pixel']['rmse'] = mapfit.accuracy.RMSE(resids)
    dct['pixel']['max'] = float(resids.max())
    # then percent (of image pixel dims)
    dct['percent'] = {}
    im = Image.open('maps/{}'.format(truth_fil))
    diag = hypot(*im.size)
    img_radius = float(diag/2.0)
    dct['percent']['rmse'] = (dct['pixel']['rmse'] / img_radius) * 100 # percent of half the max dist (from img center to corner)
    dct['percent']['max'] = (dct['pixel']['max'] / img_radius) * 100 # percent of half the max dist (from img center to corner)
    # ...

    
    ### (diff from orig controlpoints?)
    # ...


    ### all labels, for reference
    root = '_'.join(georef_root.split('_')[:4])
    allnames = pg.VectorData('maps/{}_placenames.geojson'.format(root))
    dct['labels'] = len(allnames)


    ### distribution of all labels?
    # ignore for now, bug in col row values
##    points = [(f['col'],f['row']) for f in allnames] 
##    scat = scat_metric(points)
##    dct['labels_scat_pix'] = scat
##    dct['labels_scat_perc'] = scat / float(diag/2.0) # percent of half the max dist (from img center to corner)


    ### percent of labels detected
    try:
        detected = pg.VectorData('output/{}_debug_text_toponyms.geojson'.format(georef_root))
    except:
        detected = None
    if detected:
        perc = len(detected) / float(len(allnames))
        dct['labels_detected'] = perc
    else:
        dct['labels_detected'] = None # means text/toponym detection was not performed, ie given matched control points directly


    ### percent of labels used
    gcps = pg.VectorData('output/{}_controlpoints.geojson'.format(georef_root))
    perc = len(gcps) / float(len(allnames))
    dct['labels_used'] = perc


    ### distribution of controlpoints used
    points = [(f['origx'],f['origy']) for f in gcps]
    scat = scat_metric(points)
    dct['labels_used_scat_pix'] = scat
    dct['labels_used_scat_perc'] = scat / float(diag/2.0) # percent of half the max dist (from img center to corner)
    

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
        sampling_trans = mapfit.transforms.from_json(transdict['backward'])
    georef_errorsurf_geo = mapfit.debug.error_surface_georef(sampling_trans, georef, truth, 'geographic')
    georef_errorsurf_pix = mapfit.debug.error_surface_georef(sampling_trans, georef, truth, 'pixel')
    
    # clear memory
    def delrast(r):
        del r._cached_mask
        for b in r.bands:
            del b.img, b._cached_mask
        del r
    delrast(truth)
    delrast(georef)
    gc.collect()

    # output metrics
    errdct = error_output(georef_fil, truth_fil, georef_errorsurf_geo, georef_errorsurf_pix)
    with open('output/{}_simulation_errors.json'.format(georef_root), 'w') as fobj:
        fobj.write(json.dumps(errdct))

    # visualize geographic error
    #mapp = error_vis(georef, georef_errorsurf_geo)
    #mapp.save('maps/{}_error_vis_geo.png'.format(georef_root))

    # visualize pixel error
    #mapp = error_vis(georef, georef_errorsurf_pix)
    #mapp.save('maps/{}_error_vis_pix.png'.format(georef_root))

    # clear memory
    delrast(georef_errorsurf_geo)
    delrast(georef_errorsurf_pix)
    gc.collect()
    

def run_in_process(georef_fil, truth_fil):
    georef_root,ext = os.path.splitext(georef_fil)
    logger = codecs.open('output/{}_simulation_errors_log.txt'.format(georef_root), 'w', encoding='utf8', buffering=0)
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

    maxprocs = MAXPROCS 
    procs = []

    for imfil in itermaps():
        fil_root = imfil.split('_image.')[0]
        print(fil_root)
        
            

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

        ## perfect1 topos
        exactfil = '{}_georeferenced_perfect1_topos.tif'.format(fil_root)
        if os.path.lexists('output/{}'.format(exactfil)):
            p = mp.Process(target=run_in_process,
                           args=(exactfil,imfil),
                           )
            p.start()
            procs.append(p)
            
        ## perfect2 matches
        exactfil = '{}_georeferenced_perfect2_matches.tif'.format(fil_root)
        if os.path.lexists('output/{}'.format(exactfil)):
            p = mp.Process(target=run_in_process,
                           args=(exactfil,imfil),
                           )
            p.start()
            procs.append(p)

        ## Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)
                
    # wait for last ones
    for p in procs:
        p.join()

        



