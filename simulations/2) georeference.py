
import automap as mapfit
import pythongis as pg
import PIL, PIL.Image

import os
import sys
from time import time
import datetime
import codecs
import multiprocessing as mp

# Perform the automated georeferencing

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
TEXTCOLOR = None
WARPORDER = None


###################
# FUNCTIONS
def mapfiles():
    for fil in sorted(os.listdir('maps')):
        if fil.endswith(('_image.png','_image.jpg')):
            yield fil

def georeference_auto(fil, outfil, db, source, textcolor, warp_order, priors=None):
    # EITHER automated tool
    outfil_root = os.path.splitext(outfil)[0]
    
    mapfit.automap(fil,
                   outpath=outfil,
                   db=db,
                   source=source,
                   textcolor=textcolor,
                   warp_order=warp_order,
                   debug=True,
                   priors=priors,
                   )

def georeference_exact(fil, outfil, warp_order):
    # OR use the actual coordinates for the rendered placenames (should be approx 0 error...)
    # TODO: NEEDS MORE FIXING, eg load transform from json, dont reestimate... 
    # ...
    fil_root = os.path.splitext(fil)[0]
    outfil_root = os.path.splitext(outfil)[0]
    
    print('exact georeferencing based on original placename coordinates')
    t=time()
    im = PIL.Image.open(fil)
    
    places = pg.VectorData('{}_placenames.geojson'.format(fil_root))
    tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
    # maybe transform the xy placename coordinates to the map crs (by mistake saved as wgs84)
    # ... 
    pixels,coords = zip(*tiepoints)
    (cols,rows),(xs,ys) = zip(*pixels),zip(*coords)
    trans = mapfit.transforms.Polynomial(order=warp_order)
    forward = trans.copy()
    forward.fit(cols,rows,xs,ys)
    backward = trans.copy()
    backward.fit(cols,rows,xs,ys, invert=True)

    wim,aff = mapfit.imwarp.warp(im, forward, backward) # warp
    warped = pg.RasterData(image=wim, affine=aff) # to geodata
    warped.save(outfil)
    
    #warped = mapfit.main.warp(im, '{}_georeferenced_exact.tif'.format(fil_root), tiepoints, order=warp_order)
    #gcps = [('',oc,'',mc,[]) for oc,mc in tiepoints]
    #mapfit.main.debug_warped('maps/test_georeferenced.tif', 'maps/test_debug_warp.png', gcps)

    places.rename_field('name', 'origname')
    places.rename_field('col', 'origx')
    places.rename_field('row', 'origy')
    places.rename_field('x', 'matchx')
    places.rename_field('y', 'matchy')
    places.save('{}_controlpoints.geojson'.format(outfil_root))

    print('finished exact georeferencing', time()-t)

def process_logger(func, **kwargs):
    fil = kwargs.get('fil')
    outfil = kwargs.get('outfil')
    
    fil_root = os.path.splitext(fil)[0]
    outfil_root = os.path.splitext(outfil)[0]
    logger = codecs.open('{}_log.txt'.format(outfil_root), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))
    # run it
    func(**kwargs)


####################
# RUN

if __name__ == '__main__':

    maxprocs = 2
    procs = []

    for fil in mapfiles():
        print(fil)
        fil_root = os.path.splitext(fil)[0].replace('_image', '')

        if not ('sim_1_1_' in fil_root or 'sim_1_2_' in fil_root or 'sim_1_3_' in fil_root):
            continue

        # Local testing

        ## auto
##        georeference_auto(fil='maps/{}'.format(fil),
##                          outfil='output/{}_georeferenced_auto.tif'.format(fil_root),
##                           db=r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db",
##                           source='best',
##                           textcolor=TEXTCOLOR,
##                           warp_order=WARPORDER,)

        ## auto, assuming known placenames

        # create toponyminfo from rendered placenames
##        placenames = pg.VectorData('maps/{}_placenames.geojson'.format(fil_root))
##        toponyminfo = {'type':'FeatureCollection', 'features':[]}
##        for f in placenames:
##            geoj = {'type': 'Feature',
##                    'properties': {'name':f['name']},
##                    'geometry': {'type':'Point',
##                                 'coordinates':(f['col'],f['row'])}
##                    }
##            toponyminfo['features'].append(geoj)
##
##        georeference_auto(fil='maps/{}'.format(fil),
##                          outfil='output/{}_georeferenced_known_places.tif'.format(fil_root),
##                           db=r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db",
##                           source='best',
##                           textcolor=TEXTCOLOR,
##                           warp_order=WARPORDER,
##                           priors={'toponyminfo':toponyminfo}
##                          )

        ## exact
##        georeference_exact(fil='maps/{}'.format(fil),
##                       warp_order=ORDER,)

##        continue



        # Begin process

        ## auto
##        p = mp.Process(target=process_logger,
##                       args=[georeference_auto],
##                       kwargs=dict(fil='maps/{}'.format(fil),
##                                   outfil='output/{}_georeferenced_auto.tif'.format(fil_root),
##                                   db=r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db", #"data/gazetteers.db",
##                                   source='best',
##                                   textcolor=TEXTCOLOR,
##                                   warp_order=WARPORDER,
##                                   ),
##                       )
##        p.start()
##        procs.append((p,time()))

        ## auto, assuming known placenames

        # create toponyminfo from rendered placenames
        placenames = pg.VectorData('maps/{}_placenames.geojson'.format(fil_root))
        toponyminfo = {'type':'FeatureCollection', 'features':[]}
        for f in placenames:
            geoj = {'type': 'Feature',
                    'properties': {'name':f['name']},
                    'geometry': {'type':'Point',
                                 'coordinates':(f['col'],f['row'])}
                    }
            toponyminfo['features'].append(geoj)
            
        p = mp.Process(target=process_logger,
                       args=[georeference_auto],
                       kwargs=dict(fil='maps/{}'.format(fil),
                                   outfil='output/{}_georeferenced_known_places.tif'.format(fil_root),
                                   db=r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db", #"data/gazetteers.db",
                                   source='best',
                                   textcolor=TEXTCOLOR,
                                   warp_order=WARPORDER,
                                   priors={'toponyminfo':toponyminfo}
                                   ),
                       )
        p.start()
        procs.append((p,time()))

        ## exact
##        p = mp.Process(target=process_logger,
##                       args=[georeference_exact],
##                       kwargs=dict(fil='maps/{}'.format(fil),
##                                   outfil='output/{}_georeferenced_exact.tif'.format(fil_root),
##                                   warp_order=WARPORDER,),
##                       )
##        p.start()
##        procs.append((p,time()))

        # Wait in line
        while len(procs) >= maxprocs:
            for p,t in procs:
                if not p.is_alive():
                    procs.remove((p,t))
                elif time()-t > 600:
                    p.terminate()
                    procs.remove((p,t))






