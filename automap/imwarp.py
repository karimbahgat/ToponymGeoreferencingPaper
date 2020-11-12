
import numpy as np
import PIL, PIL.Image
import math


def warp(im, transform, invtransform, resample='nearest'):
    if not im.mode == 'RGBA':
        im = im.convert('RGBA')

    # calc output bounds based on transforming source image pixel edges and diagonal distance, ala GDAL
    # TODO: alternatively based on internal grid or just all the pixels
    # see https://github.com/OSGeo/gdal/blob/60d8a9ca09c466225508cb82e30a64aefa899a41/gdal/alg/gdaltransformer.cpp#L135

    # NOTE: uses forward transform to calc output bounds, and backward transform for resampling
    # but for polynomial order >1 backward transform is reestimated on the points (inverse doesnt work)
    # and can be noticably different than the forward transform, thus miscalculating the bounds
    # TODO: maybe need a fix somehow...
    
    print 'calculating coordinate bounds'
    pixels = []
    imw,imh = im.size

    # get all pixels
    for row in range(0, imh+1): #, imh//10):
        for col in range(0, imw+1): #, imw//10):
            pixels.append((col,row))
    
##    # get top and bottom edges
##    for row in [0, imh+1]: # +1 to incl bottom of last row
##        for col in range(0, imw+1, imw//20): # +1 to incl right of last column, ca 20 points along
##            pixels.append((col,row))
##
##    # get left and right edges
##    for col in [0, imw+1]: # +1 to incl right of last column
##        for row in range(0, imh+1, imh//20): # +1 to incl bottom of last row, ca 20 points along
##            pixels.append((col,row))

    cols,rows = zip(*pixels)

    # transform and get bounds
    predx,predy = transform.predict(cols, rows)
    predx = predx[~np.isnan(predx)]
    predy = predy[~np.isnan(predy)]
    xmin,ymin,xmax,ymax = predx.min(), predy.min(), predx.max(), predy.max()

    # TODO: maybe walk along output edges and backwards transform
    # to make sure covers max possible of source img
    # in case forward/backward transform is slightly mismatched
    # ...

    # calc diagonal dist and output dims
    dx,dy = xmax-xmin, ymax-ymin
    diag = math.hypot(dx, dy)
    xyscale = diag / float(math.hypot(imw, imh))
    w,h = int(dx / xyscale), int(dy / xyscale)

##    downsize = 10
##    w = int(w/float(downsize))
##    h = int(h/float(downsize))
##    xscale = dx / float(w)
##    yscale = dy / float(h)
    
    # set affine
    xoff,yoff = xmin,ymin
    xscale = yscale = xyscale 
    if predy[0] > predy[-1]:
        yoff = ymax
        yscale *= -1
    affine = [xscale,0,xoff, 0,yscale,yoff]

    # resampling
    if resample == 'nearest':
    
        print 'backwards mapping and resampling'
        coords = []
        for row in range(h):
            y = yoff + row*yscale
            for col in range(w):
                x = xoff + col*xscale
                coords.append((x,y))
        xs,ys = zip(*coords)
        backpredx,backpredy = invtransform.predict(xs, ys)
        backpred = np.column_stack((backpredx, backpredy))
        backpred = backpred.reshape((h,w,2))
        
        print 'writing to output'
        # slow, can prob optimize even more by using direct numpy indexing
        # 4 bands, fourth is the alpha, invisible for pixels that were not sampled
        # currently assumes input image is RGBA only... 
        outarr = np.zeros((h, w, 4), dtype=np.uint8)
        imload = im.load()
        for row in range(h):
            for col in range(w):
                origcol,origrow = backpred[row,col]
                if math.isnan(origcol) or math.isnan(origrow):
                    continue
                origcol,origrow = int(math.floor(origcol)), int(math.floor(origrow))
                if 0 <= origcol < imw and 0 <= origrow < imh:
                    rgba = list(imload[origcol,origrow])
                    #rgba[-1] = 255 # fully visible
                    outarr[row,col,:] = rgba

    else:
        raise ValueError('Unknown resample arg: {}'.format(resample))

    out = PIL.Image.fromarray(outarr)
    return out, affine





    
