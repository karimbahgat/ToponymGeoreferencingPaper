import sys
sys.path.insert(0, '../dependencies/georeference maps')

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



print(os.getcwd())
try:
    os.chdir('simulations')
except:
    pass



###################
# PARAMS
TEXTCOLOR = (0,0,0) # rgb color tuple of map text, or None for autodetect
WARPORDER = None # polynomial warp order, or None for detecting optimal order
MAXPROCS = 2#20 # number of available cpu cores / parallel processes


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

    maxprocs = MAXPROCS
    procs = []

    for fil in mapfiles():
        print(fil)
        fil_root = os.path.splitext(fil)[0].replace('_image', '')
        
        #if fil_root < 'sim_9_5_':
        #    continue
        


        # Begin process

        ## auto
        p = mp.Process(target=process_logger,
                       args=[georeference_auto],
                       kwargs=dict(fil='maps/{}'.format(fil),
                                   outfil='output/{}_georeferenced_auto.tif'.format(fil_root),
                                   db="../data/gazetteers.db",
                                   source='best',
                                   textcolor=TEXTCOLOR,
                                   warp_order=WARPORDER,
                                   ),
                       )
        p.start()
        procs.append((p,time()))


        # Wait in line
        while len(procs) >= maxprocs:
            for p,t in procs:
                if not p.is_alive():
                    procs.remove((p,t))
                elif time()-t > 900:
                    p.terminate()
                    procs.remove((p,t))

                    
    # waiting for last ones
    for p,t in procs:
        p.join()





