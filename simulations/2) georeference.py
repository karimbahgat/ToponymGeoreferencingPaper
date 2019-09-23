
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
ORDER = 1


###################
# FUNCTIONS
def mapfiles():
    for fil in sorted(os.listdir('maps')):
        if fil.endswith(('_image.png','_image.jpg')):
            yield fil

def georeference_auto(fil, db, source, textcolor, warp_order):
    # EITHER automated tool
    fil_root = os.path.splitext(fil)[0].replace('_image', '')
    logger = codecs.open('{}_georeferenced_auto_log.txt'.format(fil_root), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))
    
    mapfit.automap(fil,
               outpath='{}_georeferenced_auto.tif'.format(fil_root),
               db=db,
               source=source,
               textcolor=textcolor,
               warp_order=warp_order,
               debug='ocr',
               )

def georeference_exact(fil, warp_order):
    # OR use the actual coordinates for the rendered placenames (should be approx 0 error...)
    fil_root = os.path.splitext(fil)[0].replace('_image', '')
    logger = codecs.open('{}_georeferenced_exact_log.txt'.format(fil_root), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))
    
    print('exact georeferencing based on original placename coordinates')
    t=time()
    im = PIL.Image.open(fil)
    places = pg.VectorData('{}_placenames.geojson'.format(fil_root))
    tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
    warped = mapfit.main.warp(im, '{}_georeferenced_exact.tif'.format(fil_root), tiepoints, order=warp_order)
    #gcps = [('',oc,'',mc,[]) for oc,mc in tiepoints]
    #mapfit.main.debug_warped('maps/test_georeferenced.tif', 'maps/test_debug_warp.png', gcps)
    print('finished exact georeferencing', time()-t)


####################
# RUN

if __name__ == '__main__':

    maxprocs = 4
    procs = []

    for fil in mapfiles():
        print(fil)

##        # Local testing
##
##        ## auto
##        georeference_auto(fil='maps/{}'.format(fil),
##                       db=r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db",
##                       source='best',
##                       textcolor=(0,0,0),
##                       warp_order=ORDER,)
##
##        ## exact
##        georeference_exact(fil='maps/{}'.format(fil),
##                       warp_order=ORDER,)
##
##        continue

        # Begin process

        ## auto
        p = mp.Process(target=georeference_auto,
                       kwargs=dict(fil='maps/{}'.format(fil),
                                   db="data/gazetteers.db",
                                   source='best',
                                   textcolor=(0,0,0),
                                   warp_order=ORDER,),
                       )
        p.start()
        procs.append(p)

        ## exact
        p = mp.Process(target=georeference_exact,
                       kwargs=dict(fil='maps/{}'.format(fil),
                                   warp_order=ORDER,),
                       )
        p.start()
        procs.append(p)

        # Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)






