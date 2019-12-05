
import numpy as np
import PIL, PIL.Image


def warp(im, transform, invtransform, resample='nearest'):
    if not im.mode == 'RGBA':
        im = im.convert('RGBA')
    
    pixels = []
    for row in range(im.size[1]):
        for col in range(im.size[0]):
            pixels.append((col,row))
    cols,rows = zip(*pixels)
    
    print 'calculating coordinate bounds'
    predx,predy = transform.predict(cols, rows)
    xmin,ymin,xmax,ymax = predx.min(), predy.min(), predx.max(), predy.max()
    aspect = (ymax-ymin) / float(xmax-xmin)
    w,h = int(im.size[0]), int(im.size[0]*aspect)
    xoff,yoff = xmin,ymin
    xscale = (xmax-xmin)/float(w)
    yscale = (ymax-ymin)/float(h)
    if predy[0] > predy[-1]:
        yoff = ymax
        yscale *= -1
    affine = [xscale,0,xoff, 0,yscale,yoff]
    #print w,h,affine

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
        imw,imh = im.size
        imload = im.load()
        for row in range(h):
            for col in range(w):
                origcol,origrow = backpred[row,col]
                origcol,origrow = map(int, (origcol,origrow))
                if 0 <= origcol < imw and 0 <= origrow < imh:
                    rgba = list(imload[origcol,origrow])
                    rgba[-1] = 255
                    outarr[row,col,:] = rgba

    else:
        raise ValueError('Unknown resample arg: {}'.format(resample))

    out = PIL.Image.fromarray(outarr)
    return out, affine





    
