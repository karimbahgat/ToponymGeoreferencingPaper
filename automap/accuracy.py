
import numpy as np


def residuals(inx, iny, outx, outy, distance='L1'):
    # to arrays
    inx = np.array(inx)
    iny = np.array(iny)
    outx = np.array(outx)
    outy = np.array(outy)
    
    if distance == 'L1':
        # eucledian distances
        resids = np.sqrt((inx - outx)**2 + (iny - outy)**2)
    elif distance == 'L2':
        # eucledian distances squared (larger errors penalized)
        resids = np.sqrt((inx - outx)**2 + (iny - outy)**2) ** 2
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

def RMSE(residuals):
    # root mean square error
    return (residuals**2).sum() / residuals.shape[0]

def MAE(residuals):
    # mean absolute error
    return abs(residuals).sum() / residuals.shape[0]





    
