
import numpy as np
import math
import itertools

def polynomial(order, frompoints, topoints):
    # from https://gis.stackexchange.com/questions/225931/python-gdal-how-to-obtain-dx-dy-residual-errors-for-gcps-after-gdalwarp-usage
    fromxs, fromys = zip(*frompoints)
    toxs, toys = zip(*topoints)
    points_arr = np.stack([fromxs, fromys, toxs, toys], axis=1)
    Ax_row, Ay_row, Lx_row, Ly_row = [0,1,2,3]
    
    n = len(points_arr)
    points = np.zeros((len(points_arr), 4), dtype=np.float)
    points[:, 0] = points_arr[:, Ax_row]
    points[:, 1] = points_arr[:, Ay_row]
    points[:, 2] = points_arr[:, Lx_row]
    points[:, 3] = points_arr[:, Ly_row]

    if order == 1:
        #X = a0 + a1x + a2y
        #Y = b0 + b1x + b2y
        Axy = np.zeros((len(points_arr), 3), dtype=np.float)
        Axy[:, 0] = 1
        Axy[:, 1:3] = points[:, 0:2]
        
    elif order == 2:
        #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2
        #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2
        Axy = np.zeros((len(points_arr), 6), dtype=np.float)
        Axy[:, 0] = 1 #a0
        Axy[:, 1] = points[ : , 0] # a1
        Axy[:, 2] = points[ : , 1] # a2
        Axy[:, 3] = points[ : , 0] * points[ : , 1] # ...
        Axy[:, 4] = points[ : , 0] * points[ : , 0]
        Axy[:, 5] = points[ : , 1] * points[ : , 1]
        
    elif order == 3:
        #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2 + a6x^3 + a7x^2y + a8xy^2 + a9y^3
        #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2 + b6x^3 + b7x^2y + b8xy^2 + b9y^3
        Axy = np.zeros((len(points_arr), 10), dtype=np.float)
        Axy[:, 0] = 1 #a0
        Axy[:, 1] = points[ : , 0] # a1
        Axy[:, 2] = points[ : , 1] # a2
        Axy[:, 3] = points[ : , 0] * points[ : , 1] # ...
        Axy[:, 4] = points[ : , 0] * points[ : , 0]
        Axy[:, 5] = points[ : , 1] * points[ : , 1] #
        Axy[:, 6] = points[ : , 0] * points[ : , 0] * points[ : , 0]
        Axy[:, 7] = points[ : , 0] * points[ : , 0] * points[ : , 1]
        Axy[:, 8] = points[ : , 0] * points[ : , 1] * points[ : , 1]
        Axy[:, 9] = points[ : , 1] * points[ : , 1] * points[ : , 1]

    BX = points[:, 2]
    BY = points[:, 3]

    aaa_X = np.linalg.lstsq(Axy, BX, rcond=-1)
    bbb_Y = np.linalg.lstsq(Axy, BY, rcond=-1)

    predXs = Axy.dot(aaa_X[0])
    predYs = Axy.dot(bbb_Y[0])

    V_X = points[:, 2] - np.array(predXs)
    V_Y = points[:, 3] - np.array(predYs)
    V_XY = np.sqrt(V_X**2 + V_Y**2)

    V_XY_sum_sq = V_X_sum_sq = V_Y_sum_sq = 0
    for i in range(n):
        V_XY_sum_sq += V_XY[i] ** 2
        V_X_sum_sq += V_X[i] ** 2
        V_Y_sum_sq += V_Y[i] ** 2

    mo = math.sqrt(V_XY_sum_sq/(n))
    mox = math.sqrt(V_X_sum_sq/(n))
    moy = math.sqrt(V_Y_sum_sq/(n))

    return V_X, V_Y, V_XY, mo, mox, moy, predXs, predYs, aaa_X[0], bbb_Y[0]

def predict(order, points, coeff_x, coeff_y, invert=False):
    points = np.array(points)

    if order == 1:
        #X = a0 + a1x + a2y
        #Y = b0 + b1x + b2y
        Axy = np.zeros((len(points), 3), dtype=np.float)
        Axy[:, 0] = 1
        Axy[:, 1:3] = points[:, 0:2]

    if order == 2:
        #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2
        #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2
        Axy = np.zeros((len(points), 6), dtype=np.float)
        Axy[:, 0] = 1 #a0
        Axy[:, 1] = points[ : , 0] # a1
        Axy[:, 2] = points[ : , 1] # a2
        Axy[:, 3] = points[ : , 0] * points[ : , 1] # ...
        Axy[:, 4] = points[ : , 0] * points[ : , 0]
        Axy[:, 5] = points[ : , 1] * points[ : , 1]

    elif order == 3:
        #X = a0 + a1x + a2y + a3xy + a4x^2 + a5y^2 + a6x^3 + a7x^2y + a8xy^2 + a9y^3
        #Y = b0 + b1x + b2y + b3xy + b4x^2 + b5y^2 + b6x^3 + b7x^2y + b8xy^2 + b9y^3
        Axy = np.zeros((len(points), 10), dtype=np.float)
        Axy[:, 0] = 1 #a0
        Axy[:, 1] = points[ : , 0] # a1
        Axy[:, 2] = points[ : , 1] # a2
        Axy[:, 3] = points[ : , 0] * points[ : , 1] # ...
        Axy[:, 4] = points[ : , 0] * points[ : , 0]
        Axy[:, 5] = points[ : , 1] * points[ : , 1] #
        Axy[:, 6] = points[ : , 0] * points[ : , 0] * points[ : , 0]
        Axy[:, 7] = points[ : , 0] * points[ : , 0] * points[ : , 1]
        Axy[:, 8] = points[ : , 0] * points[ : , 1] * points[ : , 1]
        Axy[:, 9] = points[ : , 1] * points[ : , 1] * points[ : , 1]

    # predict
    if invert:
        A = np.eye(len(coeff_x))
        A[1,:] = coeff_x
        A[2,:] = coeff_y
        Ainv = np.linalg.inv(A)
        coeff_x = Ainv[1,:]
        coeff_y = Ainv[2,:]

    predXs = Axy.dot(coeff_x)
    predYs = Axy.dot(coeff_y)

    # experiment with the inverse transform
    # in the case of affine, stacking coeff_x and coeff_y and 0,0,1 becomes the A matrix
    # when inverted becomes the inverted coefficients
##    print coeff_x
##    print coeff_y
##    print np.row_stack((coeff_x, coeff_y))
##    print points
##    print Axy.shape
##    inv = np.linalg.inv(Axy)
##    print predXs
##    print predYs
##    predXs = inv.dot(coeff_x)
##    predYs = inv.dot(coeff_y)
##    print predXs
##    print predYs

    # residuals
##    V_X = points[:, 2] - np.array(predXs)
##    V_Y = points[:, 3] - np.array(predYs)
##    V_XY = np.sqrt(V_X**2 + V_Y**2)

    return np.column_stack((predXs, predYs))

def optimal_rmse(order, frompoints, topoints, max_residual=0.1, min_points=3):
    # try all combinations, but not physically possible
    # ...

    # try many but not all combinations
##    min_tolerance = 0.1
##    
##    tiepoints = zip(frompoints, topoints)
##
##    best = float('inf')
##    best_frompoints = None
##    best_topoints = None
##    best_residuals = None
##    min_points = min(len(tiepoints), min_points)
##    print ''
##    for numpoints in reversed(range(min_points, len(tiepoints)+1)):
##        print 'N = %s' % numpoints, best, len(tiepoints)
##        for tiepoints_new in itertools.combinations(tiepoints, numpoints):
##            frompoints_new, topoints_new = zip(*tiepoints_new)
##            res_x, res_y, res_xy, rmse, rmse_x, rmse_y, pred_x, pred_y = polynomial(order, frompoints_new, topoints_new)
##            #print rmse
##            if rmse < best:
##                tiepoints = zip(frompoints_new, topoints_new)
##                best_frompoints = frompoints_new
##                best_topoints = topoints_new
##                best_residuals = res_xy
##                best = rmse
##                
##        if best < min_tolerance:
##            break
##
##    return best, best_frompoints, best_topoints, best_residuals

    # more effective to just drop residuals above threshold
    print ''
    frompoints = list(frompoints)
    topoints = list(topoints)
    while True:
        if len(frompoints) == min_points:
            break
        res_x, res_y, res_xy, rmse, rmse_x, rmse_y, pred_x, pred_y, coeff_x, coeff_y = polynomial(order, frompoints, topoints)
        print 'RMSE = %s' % rmse
        max_i = np.argmax(res_xy)
        if res_xy[max_i] > max_residual:
            print 'Dropping residual', res_xy[max_i]
            frompoints.pop(max_i)
            topoints.pop(max_i)
            print 'Newlen',len(frompoints),len(topoints)
        else:
            break

    # ALTERNATIVELY: drop points whose residuals are beyond one stdev of mean?
    # ...

    return rmse, frompoints, topoints, res_xy
        
        



