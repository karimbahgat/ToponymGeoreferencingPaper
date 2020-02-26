
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


###################
# PARAMS
ORDER = 1
subsamp = 10





##################
# FUNCTIONS

# Error calculation and surfaces

def sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type='geographic'):
    # two different coordsys in and out,
    # ...comparing three different transforms between in and out (two rasters and an arbitrary transform)
    # in = pixels
    # out = coords
    # TT = truth image transform (affine)
    # ET = estimated image transform (affine)
    # ET2 = alternative estimated transform (arbitrary form)

    # ST_out sample coords are given
    # the *_in pixel coordsys cannot be assumed to be equivalent
    
    # TT_in -> TT -> TT_out      (truth_forward and truth_backward)
    # ET_in -> ET -> ET_out      (not used I guess...)
    # ET2_in -> ET2 -> ET2_out   (sampling_trans = ET2-1)

    # guiding rule is that input coordsys are different, so always use output coordsys as starting point
    # ie strategy is to send sample point through arbitrary transform ET2 and then back through TT to get the error of ET2

    # args:
    # - ST_out sample coords
    # - truth rast providing TT and TT-1
    # - trans providing ET2-1

    if error_type == 'geographic':

        # geographic error
        # (goal: diff bw ET2 and TT)
        # 1) ST_out -> ET2-1 -> ET2_in   (pixel where coord was sampled)
        # 2) ET2_in -> TT -> TT_out      (coord that truly belongs to pixel)
        # 3) voila, calc distances() bw ST_out and TT_out

        sample_pxs,sample_pys = sampling_trans.predict(sample_xs, sample_ys)
        truth_xs,truth_ys = truth_forward.predict(sample_pxs, sample_pys)
        resids = mapfit.accuracy.distances(truth_xs, truth_ys, sample_xs, sample_ys, 'geodesic')
        return sample_xs, sample_ys, truth_xs, truth_ys, resids

    elif error_type == 'pixel':

        # pixel error
        # (goal: diff bw ET2 and TT)
        # 1) ST_out -> ET2-1 -> ET2_in   (pixel where coord was sampled)
        # 2) ST_out -> TT-1 -> TT_in     (pixel that truly belongs to coord)
        # 4) voila, calc distances() bw ET2_in and TT_in

        sample_pxs,sample_pys = sampling_trans.predict(sample_xs, sample_ys)
        truth_pxs,truth_pys = truth_backward.predict(sample_xs, sample_ys)
        resids = mapfit.accuracy.distances(truth_pxs, truth_pys, sample_pxs, sample_pys, 'eucledian')
        return sample_pxs, sample_pys, truth_pxs, truth_pys, resids

    else:
        raise ValueError(str(error_type))


def georef_error_surface(sampling_trans, georef, truth, error_type='geographic'):
    ### georef grid
    # ST_out sampled from georef grid
    # (requirement: ET2 input coordsys == TT input coordsys)
    out = georef.copy(shallow=True)

    # prep args
    samples = [out.cell_to_geo(col,row) for row in range(out.height) for col in range(out.width)]
    sample_xs,sample_ys = zip(*samples)
    truth_forward = mapfit.transforms.Polynomial(order=1, A=list(truth.affine))
    truth_backward = mapfit.transforms.Polynomial(order=1, A=list(~truth.affine))

    # calc errors
    sample_xs,sample_ys,truth_xs,truth_ys,errors = sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type=error_type)

    # return as raster
    errors = errors.reshape((out.height,out.width))
    out.add_band(image=Image.fromarray(errors))
    return out


def image_error_surface(sampling_trans, georef, truth, error_type='pixel'):
    ### image grid
    # ST_out sampled from image grid
    # otherwise, same exact steps
    out = truth.copy(shallow=True)

    # prep args
    samples = [out.cell_to_geo(col,row) for row in range(out.height) for col in range(out.width)]
    sample_xs,sample_ys = zip(*samples)
    truth_forward = mapfit.transforms.Polynomial(order=1, A=list(truth.affine))
    truth_backward = mapfit.transforms.Polynomial(order=1, A=list(~truth.affine))

    # calc errors
    sample_xs,sample_ys,truth_xs,truth_ys,errors = sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type=error_type)

    # return as raster
    errors = errors.reshape((out.height,out.width))
    out.add_band(image=Image.fromarray(errors))
    return out



# Visualize error surface
def error_vis(rast, error_surface):
    print('Visualizing rast overlaid with errors')
    m = pg.renderer.Map(background='white',
                        #textoptions={'font':'freesans'},
                        )
    m.add_layer(rast)
    m.add_layer(error_surface, transparency=0.2, legendoptions={'title':'Error', 'valueformat':'.0f'})
    m.zoom_bbox(*rast.bbox)
    m.add_legend({'padding':0})
    m.render_all()
    im = m.get_image()
    return im



# Error output metrics
def error_output(georef_fil, truth_fil, geographic_error_surface, pixel_error_surface):
    print('Outputting error metrics')
    georef_root,ext = os.path.splitext(georef_fil)
    dct = {}


    ### known error surface avg + stdev
    dct['image'] = {}
    # first geographic
    resids = np.array(geographic_error_surface.bands[0].img).flatten()
    dct['image']['geographic'] = mapfit.accuracy.RMSE(resids)
    # then pixels
    resids = np.array(pixel_error_surface.bands[0].img).flatten()
    dct['image']['pixels'] = mapfit.accuracy.RMSE(resids)
    # then percent (of image pixel dims)
    im = Image.open(truth_fil)
    diag = math.hypot(*im.size)
    pixperc = dct['image']['pixels'] / float(diag)
    dct['image']['percent'] = pixperc
    # ...
    

    ### controlpoint rmse
    with open('maps/{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
    # first geog and pixels
    dct['controlpoints'] = {'geographic': transdict['forward']['error'],
                            'pixels': transdict['backward']['error'],}
    # then percent (of image pixel dims)
    im = Image.open(truth_fil)
    diag = math.hypot(*im.size)
    pixperc = dct['controlpoints']['pixels'] / float(diag)
    dct['controlpoints']['percent'] = pixperc

    
    ### (diff from orig controlpoints?)
    # ...


    ### percent of labels detected
    root = '_'.join(georef_root.split('_')[:4])
    allnames = pg.VectorData('maps/{}_placenames.geojson'.format(root))
    detected = pg.VectorData('maps/{}_debug_text_toponyms.geojson'.format(root))
    perc = len(detected) / float(len(allnames))
    dct['labels_detected'] = perc


    ### percent of labels used
    allnames = pg.VectorData('maps/{}_placenames.geojson'.format(root))
    gcps = pg.VectorData('maps/{}_controlpoints.geojson'.format(root))
    perc = len(gcps) / float(len(allnames))
    dct['labels_used'] = perc
    

    return dct



# Main routine

def run_error_assessment(georef_fil, truth_fil, gcps_fil):
    georef_root,ext = os.path.splitext(georef_fil)
    logger = codecs.open('maps/{}_error_log.txt'.format(georef_root), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))

    # original/simulated map
    truth = pg.RasterData('maps/{}'.format(truth_fil))
    print truth.affine
    
    # georeferenced/transformed map
    georef = pg.RasterData('maps/{}'.format(georef_fil))
    print georef.affine

    # calc error surface
    with open('maps/{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
        sampling_trans = mapfit.transforms.from_json(transdict['backward'])
    georef_errorsurf_geo = georef_error_surface(sampling_trans, georef, truth, 'geographic')
    georef_errorsurf_pix = georef_error_surface(sampling_trans, georef, truth, 'pixel')

    # output metrics
    errdct = error_output(georef_fil, truth_fil, georef_errorsurf_geo, georef_errorsurf_pix)
    with open('maps/{}_error.json'.format(georef_root), 'w') as fobj:
        fobj.write(json.dumps(errdct))

    # visualize geographic error
    im = error_vis(georef, georef_errorsurf_geo)
    im.save('maps/{}_error_vis_geo.png'.format(georef_root))

    # visualize pixel error
    im = error_vis(georef, georef_errorsurf_pix)
    im.save('maps/{}_error_vis_pix.png'.format(georef_root))



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

        #error_assess(fil)
        #continue

        # Begin process

        ## auto
        autofil = '{}_georeferenced_auto.tif'.format(fil_root)
        print(imfil,autofil)
        if os.path.lexists('maps/{}'.format(autofil)):
            gcps = '{}_georeferenced_auto_controlpoints.geojson'.format(fil_root)
            p = mp.Process(target=run_error_assessment,
                           args=(autofil,imfil,gcps),
                           )
            p.start()
            procs.append(p)

        ## exact
        exactfil = '{}_georeferenced_exact.tif'.format(fil_root)
        if os.path.lexists('maps/{}'.format(exactfil)):
            gcps = '{}_georeferenced_exact_controlpoints.geojson'.format(fil_root)
            p = mp.Process(target=run_error_assessment,
                           args=(exactfil,imfil,gcps),
                           )
            p.start()
            procs.append(p)

        # Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)

        



