
import automap as mapfit
import os
import sys
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



###################
# PARAMS
ORDER = 1


###################
# FUNCTIONS
def mapfiles():
    for fil in os.listdir('maps'):
        if fil.endswith(('_image.png','_image.jpg')):
            yield fil

def georeference(fil, source, textcolor, warp_order):
    # EITHER automated tool
    logger = codecs.open('{}_georeferenced_log.txt'.format(fil), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))
    mapfit.automap(fil,
               source=source,
               textcolor=textcolor,
               warp_order=warp_order,
               )

    # OR use the actual coordinates for the rendered placenames (should be approx 0 error...)
    ##im = Image.open('maps/test.png')
    ##places = pg.VectorData('maps/test_placenames.geojson')
    ##tiepoints = [((f['col'],f['row']),(f['x'],f['y'])) for f in places]
    ##warped = mapfit.main.warp(im, 'maps/test_georeferenced.tif', tiepoints, order=ORDER)
    ##gcps = [('',oc,'',mc,[]) for oc,mc in tiepoints]
    ##mapfit.main.debug_warped('maps/test_georeferenced.tif', 'maps/test_debug_warp.png', cps)


####################
# RUN

if __name__ == '__main__':

    maxprocs = 8
    procs = []

    for fil in mapfiles():
        print(fil)

##        georeference(fil='maps/{}'.format(fil),
##                       source='ciesin',
##                       textcolor=(0,0,0),
##                       warp_order=ORDER,)
##        continue

        # Begin process
        p = mp.Process(target=georeference,
                       kwargs=dict(fil='maps/{}'.format(fil),
                                   source='ciesin',
                                   textcolor=(0,0,0),
                                   warp_order=ORDER,),
                       )
        p.start()
        procs.append(p)

        # Wait in line
        while len(procs) >= maxprocs:
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)






