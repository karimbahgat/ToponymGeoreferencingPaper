
import numpy as np
import math


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

##def refine_leave_one_out(inx, iny, outx, outy, metric):
##    # leave one residual out at a time, drop the one that improves accuracy the most
##    acc = metric(residuals)
##    for res in residuals:
##        resids = list(residuals)
##        resids.pop(res)
##        acc = metric(residuals)
##        ...

##def refine_residuals(residuals):
##    # or maybe based on transformer, plus all gcps...
##    # eg: forward, backward, frompoints, topoints, max_residual=0.1, min_points=3
##    x = residuals
##    def diststats(x):
##        mean = sum(x) / float(len(x))
##        sqdev = [(v-mean)**2 for v in x]
##        stdev = math.sqrt(sum(sqdev)/float(len(x)))
##        return mean,stdev
##    def dropoutliers(x, mean, stdev):
##        return [v for v in x if mean-stdev*2 < v < mean+stdev*2]
##    mean,stdev = diststats(x)
##    x = dropoutliers(x)
##    return x

def drop_outliers(transform, inpoints, outpoints, max_residual=None, geodesic=False):
    inx,iny = zip(*inpoints)
    outx,outy = zip(*outpoints)
    predx,predy = transform.predict(inx,iny)
    if geodesic:
        resids = residuals(outx,outy,predx,predy,'geodesic')
    else:
        resids = residuals(outx,outy,predx,predy)

    # calculate residual stats
    def diststats(vals):
        mean = sum(vals) / float(len(vals))
        sqdev = [(v-mean)**2 for v in vals]
        stdev = math.sqrt(sum(sqdev)/float(len(vals)))
        return mean,stdev
    mean,stdev = diststats(resids)
    
    # drop bad points with bad residuals
    inpoints_new = []
    outpoints_new = []
    for i in range(len(inpoints)):
        resid = resids[i]
        if mean-stdev*2 < resid < mean+stdev*2:
            if max_residual and resid > max_residual:
                continue
            inpoints_new.append(inpoints[i])
            outpoints_new.append(outpoints[i])
            
    return inpoints_new, outpoints_new

##def drop_gcps_stdev(resids):
##    # returns indexes of the resids to drop
##    
##    # calculate residual stats
##    def diststats(x):
##        mean = sum(x) / float(len(x))
##        sqdev = [(v-mean)**2 for v in x]
##        stdev = math.sqrt(sum(sqdev)/float(len(x)))
##        return mean,stdev
##    mean,stdev = diststats(resids)
##    
##    # drop bad points with bad residuals
##    drop = []
##    for i,resid in enumerate(resids):
##        if not mean-stdev*2 < resid < mean+stdev*2:
##            drop.append(i)
##            
##    return drop

def RMSE(residuals):
    # root mean square error
    return math.sqrt( (residuals**2).sum() / residuals.shape[0] )

def MAE(residuals):
    # mean absolute error
    return abs(residuals).sum() / residuals.shape[0]

##def LOO(residuals):
##    # leave-one-out bootstrap method to calculate error
##    pass





    
