
import numpy as np
import math


def predict_gcps(transform, inpoints, outpoints):
    inx,iny = zip(*inpoints)
    outx,outy = zip(*outpoints)
    predx,predy = transform.predict(inx,iny)
    return outpoints, zip(predx, predy)

def predict_loo(transform, inpoints, outpoints):
    # leave-one-out and reestimate bootstrap method
    predpoints = []
    for inpoint,outpoint in zip(inpoints, outpoints):
        # remove gcp and reestimate transform
        _inpoints = list(inpoints)
        _inpoints.remove(inpoint)
        _outpoints = list(outpoints)
        _outpoints.remove(outpoint)

        inx,iny = zip(*_inpoints)
        outx,outy = zip(*_outpoints)
        transform.fit(inx, iny, outx, outy)

        # calc err bw observed outpoint and predicted outpoint
        inx,iny = inpoint
        outx,outy = outpoint
        predx,predy = transform.predict([inx], [iny])
        predpoints.append((predx[0], predx[1]))
    return outpoints, predpoints

def residuals(inx, iny, outx, outy, distance='eucledian'):
    # to arrays
    inx = np.array(inx)
    iny = np.array(iny)
    outx = np.array(outx)
    outy = np.array(outy)
    
    if distance == 'eucledian':
        # eucledian distances
        resids = np.sqrt((inx - outx)**2 + (iny - outy)**2)
    elif distance == 'geodesic':
        # geodesic is geodesic distance between lat-lon coordinates
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            """
            # https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
            # convert decimal degrees to radians 
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            # haversine formula 
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = (np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a)) 
            km = 6367 * c
            return km
        resids = haversine(inx, iny, outx, outy)

    return resids

##def drop_outliers(transform, inpoints, outpoints, max_residual=None, geodesic=False):
##    inx,iny = zip(*inpoints)
##    outx,outy = zip(*outpoints)
##    predx,predy = transform.predict(inx,iny)
##    if geodesic:
##        resids = residuals(outx,outy,predx,predy,'geodesic')
##    else:
##        resids = residuals(outx,outy,predx,predy)
##
##    # calculate residual stats
##    def diststats(vals):
##        mean = sum(vals) / float(len(vals))
##        sqdev = [(v-mean)**2 for v in vals]
##        stdev = math.sqrt(sum(sqdev)/float(len(vals)))
##        return mean,stdev
##    mean,stdev = diststats(resids)
##    
##    # drop bad points with bad residuals
##    inpoints_new = []
##    outpoints_new = []
##    for i in range(len(inpoints)):
##        resid = resids[i]
##        if mean-stdev*2 < resid < mean+stdev*2:
##            if max_residual and resid > max_residual:
##                continue
##            inpoints_new.append(inpoints[i])
##            outpoints_new.append(outpoints[i])
##            
##    return inpoints_new, outpoints_new


# metrics

def RMSE(residuals):
    return math.sqrt( (residuals**2).sum() / float(residuals.shape[0]) )

def MAE(residuals):
    return abs(residuals).sum() / float(residuals.shape[0])

##def RMSE(transform, inpoints, outpoints):
##    # root mean square error
##    inx,iny = zip(*inpoints)
##    outx,outy = zip(*outpoints)
##    predx,predy = transform.predict(inx,iny)
##    resids = residuals(outx, outy, predx, predy)
##    err = math.sqrt( (residuals**2).sum() / residuals.shape[0] )
##    return err, resids

##def MAE(transform, inpoints, outpoints):
##    # mean absolute error
##    inx,iny = zip(*inpoints)
##    outx,outy = zip(*outpoints)
##    predx,predy = transform.predict(inx,iny)
##    resids = residuals(outx, outy, predx, predy)
##    err = abs(residuals).sum() / residuals.shape[0]
##    return err, resids






    
