
from . import segmentation

import numpy as np

import PIL, PIL.Image

import cv2

import shapely



def filter_toponym_candidates(data, seginfo=None):
    if seginfo:
        # inclusion region
        incl_shp = None
        mapregion = next((f['geometry'] for f in seginfo['features'] if f['properties']['type'] == 'Map'), None)
        if mapregion:
            mapshp = shapely.geometry.asShape(mapregion)
            incl_shp = mapshp

        # exclusion region
        excl_shp = None
        boxes = [f['geometry'] for f in seginfo['features'] if f['properties']['type'] == 'Box']
        if boxes:
            boxshps = [shapely.geometry.asShape(box) for box in boxes]
            excl_shp = shapely.ops.unary_union(boxshps)
    
    topotexts = []
    for text in data:
        # only texts that resemble a toponym
        alphachars = text['text_alphas']
        if len(alphachars) < 2:
            # toponyms must contain at least 2 alpha chars
            continue
        if any((ch.isnumeric() for ch in text['text_clean'])):
            # cannot contain any numbers
            continue
        if not text['text_clean'][0].isupper():
            # first char must be uppercase
            continue
        if len([ch for ch in alphachars if ch.isupper()]) > (len(alphachars) / 2.0):
            # are not all uppercase
            # upper = more than half of characters is uppercase (to allow for minor ocr upper/lower errors)
            continue

        # only texts in relevant parts of the image
        if seginfo and (incl_shp or excl_shp):
            bbox = [text['left'], text['top'], text['left']+text['width'], text['top']+text['height']]
            text_shp = shapely.geometry.box(*bbox)
            # must be in inclusion region
            if incl_shp and not text_shp.intersects(incl_shp):
                continue
            # must not be in exclusion region
            if excl_shp and text_shp.intersects(excl_shp):
                continue
        
        topotexts.append(text)
    return topotexts

def detect_toponym_anchors(im, data, debug=False):
##    lab = segmentation.rgb_to_lab(im)
##    l,a,b = lab.split()
##    
##    im_arr = 255 * (np.array(l) > (255/2)) # binarize on higher luminance values
##    im_arr = im_arr.astype(np.uint8)
##    #PIL.Image.fromarray(im_arr).show()

    im_arr = np.array(im).astype(np.uint8)
    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
    for r in data:
        #if not r['placename']:
        #    continue
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h

        # possibly, narrow down the actual bbox via avg pixel coords
##        onxs,onys = np.nonzero(im_arr == 0)
##        tot = len(onxs)
##        onxcounts = zip(*onxs.unique(return_counts=True))
##        onycounts = zip(*onys.unique(return_counts=True))
##        # ...hmm...

        buff = int(h * 1.5)
        filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff] = False # look in buffer zone around text box
        #filt_im_arr[y1:y2, x1:x2] = True # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
    im_arr[filt_im_arr] = 255
    #im_arr[im_arr < 255] = 0
    if debug:
        PIL.Image.fromarray(im_arr).show()

    # determine kernel size from avg text height
    h = sum([r['fontheight'] for r in data]) / len(data)
    h /= 2
    print 'kernel size', h

    # get average value in neighbourhood
##    kernel = np.ones((h,h)) / (h*h)
##    im_arr = 255 - im_arr # invert
##    im_arr = cv2.filter2D(im_arr, -1, kernel) #cv2.blur(im_arr, (buff,buff)) #cv2.boxFilter(im_arr, -1, buff)
##    #im_arr = cv2.distanceTransform(im_arr, cv2.DIST_L1, 3)
##    print im_arr.max(), im_arr.mean()
##    PIL.Image.fromarray(im_arr).show()

    # get distance to center
    im_arr = 255 - im_arr # invert
    ret,im_arr = cv2.threshold(im_arr,200,255,cv2.THRESH_BINARY)
    if debug:
        PIL.Image.fromarray(im_arr).show()
    im_arr = cv2.distanceTransform(im_arr, cv2.DIST_L2, 3)
    dist_arr = im_arr.copy()
    print im_arr.max(), im_arr.mean()
    if debug:
        PIL.Image.fromarray(im_arr*50).show()

    # get highest value in neighbourhood
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(h*2,h*2))
    max_arr = cv2.dilate(im_arr,kernel,iterations=1)
    if debug:
        PIL.Image.fromarray(max_arr*50).show()

    im_arr[im_arr != max_arr] = 0
    #im_arr[im_arr < 100] = 0
    #im_arr[im_arr == 0] = 0
    #im_arr[(im_arr > 0) & (im_arr == max_arr)] = 255
    print im_arr.max(), im_arr.mean()
    if debug:
        PIL.Image.fromarray(im_arr*50).show()

    # get shape centers
    centers = np.nonzero(im_arr)
    centervals = list(im_arr[centers]) #[im_arr[cx, cy] for cx,cy in centers]
    centers = [(pt[1],pt[0]) for pt in np.transpose(centers)]
    print 'centers', len(centers)

    # link each text to closest point
    if debug:
        import pyagg
        c = pyagg.canvas.from_image(PIL.Image.fromarray(dist_arr*50))
    
    points = []
    for r in data:
        #if not r['placename']:
        #    continue
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        # buffer
        x1 -= h
        x2 += h
        y1 -= h
        y2 += h
        # first those within buffered bbox
        nearby = filter(lambda(x,y): x1 < x < x2 and y1 < y < y2, centers)

        if debug:
            c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor=(0,255,0))

        if nearby:
            # choose the nearest circle
##            nearest = sorted(nearby, key=lambda x: x[1])[0]
##            c = nearest[0]
##            print text,c
##            p = (int(c.centroid.x), int(c.centroid.y))
##            points.append((text, p))
            
            # or choose the highest value pixel (most concentrated)
            maxpt = max(nearby, key=lambda pt: centervals[centers.index(pt)])
            maxpt = map(int, maxpt)
            r['anchor'] = maxpt

            if debug:
                for n in nearby:
                    c.draw_circle(xy=n, fillsize=centervals[centers.index(n)], fillcolor=None, outlinecolor=(255,0,0))
                c.draw_circle(xy=maxpt, fillsize=1, fillcolor=None, outlinecolor=(0,0,255))

    if debug:
        c.get_image().show()
    
    return data
