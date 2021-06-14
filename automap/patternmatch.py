# new version taking over for shapematch.py

import numpy as np


def normalize(pointset):
    # normal calc
    xs,ys = zip(*pointset)

    xmin,ymin = min(xs),min(ys)
    xmax,ymax = max(xs),max(ys)
    w,h = xmax-xmin, ymax-ymin
    scale = max(w,h)
        
    xs = [(x-xmin)/float(scale) for x in xs]
    ys = [(y-ymin)/float(scale) for y in ys]
    pointset = np.array( list(zip(xs,ys)) )

    # numpy calc
##    pointset = np.array(pointset)
##    xs,ys = pointset[:,0], pointset[:,1]
##    xmin,ymin = xs.min(),ys.min()
##    xmax,ymax = xs.max(),ys.max()
##    w,h = xmax-xmin, ymax-ymin
##    scale = max(w,h)
##        
##    xs = (xs-xmin)/float(scale)
##    ys = (ys-ymin)/float(scale)
##    pointset = np.stack([xs,ys], axis=1)
    
    return pointset

def prep_pool(pool):
    normed = []
    for feat in pool:
        norm = normalize(feat['geometry']['coordinates'])
        normed.append( (feat,norm) )
    return normed

def find_best_matches(test, pool):
    # prep test
    testnorm = normalize(test['geometry']['coordinates'])

    # calculate diffs
    results = []
    for test2,test2norm in pool:
        diff,diffs = pointset_diff(testnorm, test2norm)
        results.append( (test2,diff,diffs) )

    # return best
    return sorted(results, key=lambda pair: pair[1])

def pointset_diff(pointset1, pointset2):
    dx,dy = pointset1[:,0]-pointset2[:,0], pointset1[:,1]-pointset2[:,1]
    dists = np.hypot(dx, dy)
    diff = dists.mean()#.max()
    return diff,dists


