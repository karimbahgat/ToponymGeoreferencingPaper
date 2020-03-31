
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
            mapshp = mapshp if mapshp.is_valid else mapshp.buffer(0) # try to fix if invalid
            mapshp = mapshp if mapshp.is_valid else None # only keep if valid
            if mapshp:
                incl_shp = mapshp

        # exclusion region
        excl_shp = None
        boxes = [f['geometry'] for f in seginfo['features'] if f['properties']['type'] == 'Box']
        if boxes:
            boxshps = [shapely.geometry.asShape(box) for box in boxes]
            boxshps = [b if b.is_valid else b.buffer(0) # try to fix invalid ones
                       for b in boxshps]
            boxshps = [b for b in boxshps if b.is_valid] # only keep those that remain valid
            if len(boxshps) > 1:
                excl_shp = shapely.ops.unary_union(boxshps)
            elif len(boxshps) == 1:
                excl_shp = boxshps[0]
    
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


def detect_toponym_anchors(im, texts, toponyms, debug=False):
    '''Detect anchor points from image, set each text's anchor point with the 'anchor' key.
    - im is the original image.
    - texts is list of tesseract text dict results (used to exclude areas when looking for anchor points).
    - toponyms is those text dicts considered to be possible toponyms.
    '''
    # first threshold img to black pixels only
    # (anchors are usually thick and almost always black, so not as affected by color blending as text)
    diff = segmentation.color_difference(segmentation.quantize(im), (0,0,0))
    diff[diff > 25] = 255
    diff[diff <= 25] = 0
    anchor_im = PIL.Image.fromarray(diff)

    # OR get color changes/edges
    #changes = segmentation.color_changes(im)
    #changes[changes > 10] = 255
    #anchor_im = PIL.Image.fromarray(changes)

    # blank out all text regions
    im_arr = np.array(anchor_im).astype(np.uint8)
    im_arr_orig = im_arr.copy()
    for r in texts:
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        fh = int(r['fontheight'])

        # do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
        im_arr[y1:y2, x1:x2] = 255

    # loop texts and process each individually
    newdata = []
    for r in toponyms:
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        fh = int(r['fontheight'])

##        debug = ('Bouna' in r['text'] or \
##                 'Tchamba' in r['text'] or \
##                 'Orodaro' in r['text'] or \
##                 'Yendi' in r['text'])
        
        # extract subimg from buffer around text
        buff = int(fh * 1)
        edge = int(fh * 1)
        buff_im_arr = im_arr[y1-buff-edge:y2+buff+edge, x1-buff-edge:x2+buff+edge] #im_arr[filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff]]
        
        # look for distance anchor
        newr = detect_text_anchor_distance(buff_im_arr, r, debug=debug)

        # otherwise, look for contour anchor
        if 'anchor' not in newr:
            print 'distance failed, trying contour', r['text']
            newr = detect_text_anchor_contour(buff_im_arr, r, debug=debug)

        # add
        newdata.append(newr)

    return newdata
        
def detect_text_anchor_distance(textimg, textdata, debug=False):
    # get basic textinfo
    textdata = textdata.copy()
    x1,y1,w,h = [int(textdata[k]) for k in 'left top width height'.split()]
    x2 = x1+w
    y2 = y1+h
    fh = int(textdata['fontheight'])

    buff = int(fh * 1)
    edge = int(fh * 1)

    # prep for morphology processing
    textimg_orig = textimg.copy()
    textimg = 255 - textimg # invert

    #########
    if debug:
        import pyagg
        print 'text',textdata['text'],w,h,fh,textimg.shape
        c = pyagg.canvas.from_image(PIL.Image.fromarray(textimg_orig))
        c.custom_space(x1-buff-edge, y1-buff-edge, x2+buff+edge, y2+buff+edge)
        c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor='green', outlinewidth='1px')

    # apply distance transform
    dist_im_arr = cv2.distanceTransform(textimg, cv2.DIST_L2, 3)
    dist_im_arr = dist_im_arr[edge:-edge, edge:-edge] # remove edge !!!! 
    if debug:
        print 'raw dists',dist_im_arr.max()
        PIL.Image.fromarray(dist_im_arr*50).show()
    
    # only find actual anchors of significant size
    # so limit to centers larger than 1/4th font height, but smaller than 1x font height
    ch = fh/2.0 # value of center the size of fontsize will be half the fontsize
    ch_lower = ch/4.0
    ch_upper = ch
    dist_im_arr[dist_im_arr < ch_lower] = 0
    dist_im_arr[dist_im_arr > ch_upper] = 0
    if debug:
        print 'limit to dist range', ch_lower, ch_upper

    # get highest value ie centerpoints in neighbourhood of size fontheight
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fh,fh))
    max_arr = cv2.dilate(dist_im_arr, kernel, iterations=1) # max value within kernel neighbourhood
    dist_im_arr[dist_im_arr != max_arr] = 0
    #if debug:
    #    PIL.Image.fromarray(dist_im_arr*50).show()

    # get shape centers
    centers = np.nonzero(dist_im_arr)
    centervals = list(dist_im_arr[centers]) 
    centers = [(pt[1],pt[0]) for pt in np.transpose(centers)]
    centers = [(x1-buff+cx, y1-buff+cy) for cx,cy in centers] # offset to total img coords
    if debug:
        print 'final dists',dist_im_arr.max(),sorted(centervals, reverse=True)

    # choose the highest value pixel (most concentrated)
    if centers: 
        maxpt = max(centers, key=lambda pt: centervals[centers.index(pt)])
        maxpt = map(int, maxpt)
        textdata['anchor'] = maxpt

        if debug:
            print 'found', maxpt, max(centervals)
            for n in centers:
                c.draw_circle(xy=n, fillsize=centervals[centers.index(n)], fillcolor=None, outlinecolor=(0,0,255), outlinewidth='0.5px')
            c.draw_circle(xy=maxpt, fillsize='3px', fillcolor=None, outlinecolor=(255,0,0), outlinewidth='0.5px')

    if debug:
        c.get_image().show()

    return textdata

def detect_text_anchor_contour(textimg, textdata, debug=False):
    # get basic textinfo
    textdata = textdata.copy()
    x1,y1,w,h = [int(textdata[k]) for k in 'left top width height'.split()]
    x2 = x1+w
    y2 = y1+h
    fh = int(textdata['fontheight'])

    buff = int(fh * 1)
    edge = int(fh * 1)

    # prep for morphology processing
    textimg_orig = textimg.copy()
    textimg = 255 - textimg # invert

    if debug:
        PIL.Image.fromarray(textimg).show()

    # merge unrelated text strings plus fill in symbol holes+gaps (sixth of fontsize), via dilation-erosion (morphology closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #(max(1,fh//2),max(1,fh//2)))
    textimg = cv2.morphologyEx(textimg, cv2.MORPH_CLOSE, kernel, iterations=fh//5)

    if debug:
        PIL.Image.fromarray(textimg).show()

    # remove noise + touching lines (eight of fontsize), via erosion->dilation  (morphology opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #(max(1,fh//5),max(1,fh//5)))
    textimg = cv2.morphologyEx(textimg, cv2.MORPH_OPEN, kernel, iterations=max(1,fh//8))

    if debug:
        PIL.Image.fromarray(textimg).show()

    #########
    if debug:
        import pyagg
        print 'text',r['text'],w,h,fh,textimg.shape
        c = pyagg.canvas.from_image(PIL.Image.fromarray(textimg_orig))
        c.custom_space(x1-buff-edge, y1-buff-edge, x2+buff+edge, y2+buff+edge)
        c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor='green', outlinewidth='1px')

    # get outline contours
    contours,_ = cv2.findContours(textimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # only find contours of significant size
    # so limit to contours larger than 1/4th font height, but smaller than 1.5x font height
    ch_lower = fh/4.0
    ch_upper = fh*1.5
    if debug:
        print 'limit to dist range', ch_lower, ch_upper
    candidates = []
    for cnt in contours:
        # make sure significant size and roughly rectangular shape bbox
        x,y,w,h = cv2.boundingRect(cnt)
        if debug:
            print 'size', (w,h)
        if not (ch_lower <= w <= ch_upper and ch_lower <= h <= ch_upper and min(w,h)/float(max(w,h))) >= 0.75:
            continue
    
        # make sure is an areal shape, not a line
        area = cv2.contourArea(cnt)
        fillrate = area / float(w*h)
        if debug:
            print 'area', area, fillrate
        if not fillrate >= 0.20:
            continue

        # get shape center
        pt = x+w/2.0, y+h/2.0
        pt = x1-buff-edge+pt[0], y1-buff-edge+pt[1] # offset to total img coords
        pt = map(int, pt)

        # ensure within pure buffer
        if not (x1-buff <= pt[0] <= x2+buff and y1-buff <= pt[1] <= y2+buff):
            continue

        # ensure not inside textbox
        if (x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2):
            continue

        if debug:
            print 'candidate anchor at',pt
        candidates.append((cnt,pt))

    if candidates:
        # choose the largest contour as anchor
        cnt,pt = sorted(candidates, key=lambda(cnt,pt): cv2.contourArea(cnt))[-1]
        textdata['anchor'] = pt

        if debug:
            print 'final anchor at', pt
            for _cnt,_pt in candidates:
                c.draw_circle(xy=_pt, fillsize='3px', fillcolor=None, outlinecolor=(0,0,255), outlinewidth='0.5px')
            c.draw_circle(xy=pt, fillsize='5px', fillcolor=None, outlinecolor=(255,0,0), outlinewidth='1px')

    if debug:
        c.get_image().show()

    return textdata







########
# OLD WORKING

##def detect_toponym_anchors_distance(im, data, debug=False):
##    '''Detect anchor points via distance method, set each text's anchor point with the 'anchor' key.
##    - im must already be grayscale of possible anchor pixels.
##    - data is list of tesseract text dict results, preferably already filtered to toponyms. 
##    '''
##    #debug = True
##
##    # blank out all text regions
##    im_arr = np.array(im).astype(np.uint8)
##    im_arr_orig = im_arr.copy()
##    for r in data:
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
##        # do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##        im_arr[y1:y2, x1:x2] = 255
##
##    # find centers of distance gravity around each text buffer       
##    for r in data:
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##        
##        # extract subimg from buffer around text
##        buff = int(fh * 1)
##        edge = int(fh * 0.5)
##        buff_im_arr = im_arr[y1-buff-edge:y2+buff+edge, x1-buff-edge:x2+buff+edge] #im_arr[filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff]]
##        thresh = 25
##        buff_im_arr[buff_im_arr < thresh] = 0 # binarize
##        buff_im_arr[buff_im_arr >= thresh] = 255 # binarize
##        buff_im_arr = 255 - buff_im_arr # invert
##
##        if debug:
##            import pyagg
##            print 'text',r['text'],w,h,buff_im_arr.shape
##            c = pyagg.canvas.from_image(PIL.Image.fromarray(im_arr_orig[y1-buff:y2+buff, x1-buff:x2+buff]))
##            c.custom_space(x1-buff, y1-buff, x2+buff, y2+buff)
##            c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor='green', outlinewidth='1px')
##
##        # handle hollow shapes by filling them (third of fontsize), using morphology closing
##        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fh//3,fh//3))
##        buff_im_arr = cv2.morphologyEx(buff_im_arr, cv2.MORPH_CLOSE, kernel)
##
##        # calc distance
##        dist_im_arr = cv2.distanceTransform(buff_im_arr, cv2.DIST_L2, 3)
##        dist_im_arr = dist_im_arr[edge:-edge, edge:-edge] # remove edge !!!! 
##        if debug:
##            print 'raw dists',dist_im_arr.max()
##            PIL.Image.fromarray(dist_im_arr*50).show()
##        
##        # only find actual anchors of significant size
##        # so limit to centers larger than 1/4th font height, but smaller than 1x font height
##        ch = fh/2.0 # value of center the size of fontsize will be half the fontsize
##        ch_lower = ch/4.0
##        ch_upper = ch
##        dist_im_arr[dist_im_arr < ch_lower] = 0
##        dist_im_arr[dist_im_arr > ch_upper] = 0
##        if debug:
##            print 'limit to dist range', ch_lower, ch_upper
##
##        # get highest value ie centerpoints in neighbourhood of size fontheight
##        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fh,fh))
##        max_arr = cv2.dilate(dist_im_arr, kernel, iterations=1) # max value within kernel neighbourhood
##        dist_im_arr[dist_im_arr != max_arr] = 0
##        #if debug:
##        #    PIL.Image.fromarray(dist_im_arr*50).show()
##
##        # get shape centers
##        centers = np.nonzero(dist_im_arr)
##        centervals = list(dist_im_arr[centers]) 
##        centers = [(pt[1],pt[0]) for pt in np.transpose(centers)]
##        centers = [(x1-buff+cx, y1-buff+cy) for cx,cy in centers] # offset to total img coords
##        if debug:
##            print 'final dists',dist_im_arr.max(),sorted(centervals, reverse=True)
##
##        # choose the highest value pixel (most concentrated)
##        if centers: 
##            maxpt = max(centers, key=lambda pt: centervals[centers.index(pt)])
##            maxpt = map(int, maxpt)
##            r['anchor'] = maxpt
##
##            if debug:
##                print 'found', maxpt, max(centervals)
##                for n in centers:
##                    c.draw_circle(xy=n, fillsize=centervals[centers.index(n)], fillcolor=None, outlinecolor=(0,0,255), outlinewidth='0.5px')
##                c.draw_circle(xy=maxpt, fillsize='3px', fillcolor=None, outlinecolor=(255,0,0), outlinewidth='0.5px')
##
##        if debug:
##            c.get_image().show()
##
##    return data
##
##def detect_toponym_anchors_contour(im, data, debug=False):
##    '''Detect anchor points via contour method, set each text's anchor point with the 'anchor' key.
##    - im must already be BINARY of possible anchor pixels.
##    - data is list of tesseract text dict results, preferably already filtered to toponyms. 
##    '''
##    #debug = True
##
##    # blank out all text regions
##    im_arr = np.array(im).astype(np.uint8)
##    im_arr_orig = im_arr.copy()
##    for r in data:
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
##        # do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##        #im_arr[y1:y2, x1:x2] = 255
##
##    # find contours around each text buffer       
##    for r in data:
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
####        debug = ('Bouna' in r['text'] or \
####                 'Tchamba' in r['text'] or \
####                 'Orodaro' in r['text'] or \
####                 'Yendi' in r['text'])
##        
##        # extract subimg from buffer around text
##        buff = int(fh * 1)
##        edge = int(fh * 1)
##        buff_im_arr = im_arr[y1-buff-edge:y2+buff+edge, x1-buff-edge:x2+buff+edge] #im_arr[filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff]]
##        buff_im_arr = 255 - buff_im_arr # invert
##
##        # MAYBE ONLY DO THESE MORPHOLOGY OPS IN SECOND ROUND IF RAW CONTOURS FAIL? 
##        # ...
##
##        if debug:
##            PIL.Image.fromarray(buff_im_arr).show()
##
##        # merge unrelated text strings plus fill in symbol holes+gaps (sixth of fontsize), via dilation-erosion (morphology closing)
##        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #(max(1,fh//2),max(1,fh//2)))
##        buff_im_arr = cv2.morphologyEx(buff_im_arr, cv2.MORPH_CLOSE, kernel, iterations=fh//5)
##
##        if debug:
##            PIL.Image.fromarray(buff_im_arr).show()
##
##        # remove noise + touching lines (eight of fontsize), via erosion->dilation  (morphology opening)
##        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #(max(1,fh//5),max(1,fh//5)))
##        buff_im_arr = cv2.morphologyEx(buff_im_arr, cv2.MORPH_OPEN, kernel, iterations=max(1,fh//8))
##
##        if debug:
##            PIL.Image.fromarray(buff_im_arr).show()
##
##        #########
##        if debug:
##            import pyagg
##            print 'text',r['text'],w,h,fh,buff_im_arr.shape
##            c = pyagg.canvas.from_image(PIL.Image.fromarray(im_arr_orig[y1-buff-edge:y2+buff+edge, x1-buff-edge:x2+buff+edge]))
##            c.custom_space(x1-buff-edge, y1-buff-edge, x2+buff+edge, y2+buff+edge)
##            c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor='green', outlinewidth='1px')
##
##        # get outline contours
##        contours,_ = cv2.findContours(buff_im_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
####        if debug:
####            import pythongis as pg
####            contours_geoj = [{'type': 'Polygon',
####                              'coordinates': [ [tuple(p[0].tolist()) for p in cnt] ]}
####                              for cnt in contours]
####            contourdata = pg.VectorData()
####            for geoj in contours_geoj:
####                contourdata.add_feature([], geoj)
####            contourdata.view(fillcolor='red', fillalpha=127)
##
##        # only find contours of significant size
##        # so limit to contours larger than 1/4th font height, but smaller than 1.5x font height
##        ch_lower = fh/4.0
##        ch_upper = fh*1.5
##        if debug:
##            print 'limit to dist range', ch_lower, ch_upper
##        candidates = []
##        for cnt in contours:
##            # make sure significant size and roughly rectangular shape bbox
##            x,y,w,h = cv2.boundingRect(cnt)
##            if debug:
##                print 'size', (w,h)
##            if not (ch_lower <= w <= ch_upper and ch_lower <= h <= ch_upper and min(w,h)/float(max(w,h))) >= 0.75:
##                continue
##        
##            # make sure is an areal shape, not a line
##            area = cv2.contourArea(cnt)
##            fillrate = area / float(w*h)
##            if debug:
##                print 'area', area, fillrate
##            if not fillrate >= 0.20:
##                continue
##
##            # get shape center
##            pt = x+w/2.0, y+h/2.0
##            pt = x1-buff-edge+pt[0], y1-buff-edge+pt[1] # offset to total img coords
##            pt = map(int, pt)
##
##            # ensure within pure buffer
##            if not (x1-buff <= pt[0] <= x2+buff and y1-buff <= pt[1] <= y2+buff):
##                continue
##
##            # ensure not inside textbox
##            if (x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2):
##                continue
##
##            if debug:
##                print 'candidate anchor at',pt
##            candidates.append((cnt,pt))
##
##        if candidates:
##            # choose the largest contour as anchor
##            cnt,pt = sorted(candidates, key=lambda(cnt,pt): cv2.contourArea(cnt))[-1]
##            r['anchor'] = pt
##
##            if debug:
##                print 'final anchor at', pt
####                candidates_geoj = [{'type': 'Polygon',
####                                  'coordinates': [ [tuple(p[0].tolist()) for p in cnt] ]}
####                                  for cnt,pt in candidates]
####                for geoj in candidates_geoj:
####                    geoj['coordinates'] = [ [(x1-buff-edge+xy[0], y1-buff-edge+xy[1]) for xy in geoj['coordinates'][0]] ] # offset
####                    c.draw_geojson(geoj, fillcolor=None, outlinecolor=(0,0,255), outlinewidth='0.5px')
##                for _cnt,_pt in candidates:
##                    c.draw_circle(xy=_pt, fillsize='3px', fillcolor=None, outlinecolor=(0,0,255), outlinewidth='0.5px')
##                c.draw_circle(xy=pt, fillsize='5px', fillcolor=None, outlinecolor=(255,0,0), outlinewidth='1px')
##
##        if debug:
##            c.get_image().show()
##
##    return data

##def detect_toponym_anchors_template_contours(im, data, templates, debug=False):
##    '''Detect anchor points via template matching method, set each text's anchor point with the 'anchor' key.
##    - im must already be grayscale of possible anchor pixels. 
##    - data is list of tesseract text dict results, preferably already filtered to toponyms.
##    - templates is list of 0-1 normalized geojson vector shapes to detect. 
##    '''
##    raise NotImplementedError
##    
##    debug = True
##
##    # filter grayscale image to buffer region around each text
##    im_arr = np.array(im).astype(np.uint8)
##    im_arr_orig = im_arr.copy()
##    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
##    for r in data:
##        print r
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
##        # do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##        filt_im_arr[y1:y2, x1:x2] = True 
##
##        # extract subimg from buffer around text
##        buff = int(fh * 1)
##        buff_im_arr = im_arr[y1-buff:y2+buff, x1-buff:x2+buff] #im_arr[filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff]]
##        thresh = 25
##        buff_im_arr[buff_im_arr < thresh] = 0 # binarize
##        buff_im_arr[buff_im_arr >= thresh] = 255 # binarize
##        buff_im_arr = 255 - buff_im_arr # invert
##        if debug:
##            PIL.Image.fromarray(buff_im_arr).show()
##
##        # extract contours from img
##        contours,_ = cv2.findContours(buff_im_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
##        contours_geoj = [{'type': 'Polygon',
##                          'coordinates': [ [tuple(p[0].tolist()) for p in cnt] ]}
##                          for cnt in contours]
##        
##        # anchor contours must be of significant size
##        # so limit to centers larger than 1/4th font height, but smaller than 1x font height
####        ch = fh/2.0 # value of center the size of fontsize will be half the fontsize
####        ch_lower = ch/4.0
####        ch_upper = ch
####        im_arr[im_arr < ch_lower] = 0
####        im_arr[im_arr > ch_upper] = 0
####        print 'limit to dist range', ch_lower, ch_upper
##
##        # debug
##        if debug:
##            import pythongis as pg
##            contourdata = pg.VectorData()
##            for geoj in contours_geoj:
##                contourdata.add_feature([], geoj)
##            contourdata.view(fillcolor='red', fillalpha=127)
##
##        # run vector/contour shape matching against input templates
##        # ...

##def detect_toponym_anchors_template_images(im, data, debug=False):
##    '''Detect anchor points via template matching method, set each text's anchor point with the 'anchor' key.
##    - im must already be grayscale of possible anchor pixels. 
##    - data is list of tesseract text dict results, preferably already filtered to toponyms.
##    '''
##    raise NotImplementedError
##
##    debug = True
##
##    # blank out all text regions
##    im_arr_orig = np.array(im)
##    im_arr =  np.ones(im_arr_orig.shape) * 255
##    for r in data:
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
##        # look inside buffer region
##        buff = int(fh * 1)
##        im_arr[y1-buff:y2+buff, x1-buff:x2+buff] = im_arr_orig[y1-buff:y2+buff, x1-buff:x2+buff]
##
##        # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##        im_arr[y1:y2, x1:x2] = 255
##
##    # find anchor candidates within text buffers
##    template_candidates = []
##    for r in data:
##        print 'finding template candidates near', r['text']
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##        
##        # extract subimg from buffer around text
##        buff = int(fh * 1)
##        buff_im_arr = im_arr[y1-buff:y2+buff, x1-buff:x2+buff] #im_arr[filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff]]
##        thresh = 25
##        bin_im_arr = buff_im_arr.astype(np.uint8)
##        bin_im_arr[bin_im_arr < thresh] = 0 # binarize
##        bin_im_arr[bin_im_arr >= thresh] = 255 # binarize
##        bin_im_arr = 255 - bin_im_arr # invert
##
##        # extract contours from img
##        contours,_ = cv2.findContours(bin_im_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
##        contours_geoj = [{'type': 'Polygon',
##                          'coordinates': [ [tuple(p[0].tolist()) for p in cnt] ]}
##                          for cnt in contours]
##        
##        # only find templates of significant size
##        # so limit to centers larger than 1/4th font height, but smaller than 1x font height
##        ch = fh/2.0 # value of center the size of fontsize will be half the fontsize
##        ch_lower = ch/4.0
##        ch_upper = ch
##        for geoj in contours_geoj:
##            coords = geoj['coordinates'][0]
##            xs,ys = zip(*coords)
##            xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs),max(ys)
##            w,h = xmax-xmin, ymax-ymin
##            if ch_lower <= w <= ch_upper and ch_lower <= h <= ch_upper and 0.75 < (w/float(h)) < 1.25:
##                templ = buff_im_arr[ymin:ymax+1, xmin:xmax+1].copy()
##                templ[templ == 255] = thresh
##                templ = thresh - templ # invert
##                template_candidates.append(templ)
##    print 'total candidates', len(template_candidates)
##
##    # group templates
##    print 'grouping templates'
##    grouped_template_candidates = []
##    hs,ws = zip(*[templ.shape for templ in template_candidates])
##    maxw,maxh = max(ws),max(hs)
##    def mse(imageA, imageB):
##        print 'mse',imageA.shape,'vs',imageB.shape
##        # first make same dimensions to allow comparison
##        hs,ws = zip(imageA.shape, imageB.shape)
##        maxw,maxh = max(ws),max(hs)
##        # A
##        frame = np.zeros((maxh,maxw))
##        h,w = imageA.shape
##        xoff = (maxw-w) // 2
##        yoff = (maxh-h) // 2
##        frame[yoff:yoff+h, xoff:xoff+w] = imageA
##        imageA = frame
##        print 'A',imageA
##        # B
##        frame = np.zeros((maxh,maxw))
##        h,w = imageB.shape
##        xoff = (maxw-w) // 2
##        yoff = (maxh-h) // 2
##        frame[yoff:yoff+h, xoff:xoff+w] = imageB
##        imageB = frame
##        print 'B',imageB
##        # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
##	# the 'Mean Squared Error' between the two images is the
##	# sum of the squared difference between the two images;
##	# NOTE: the two images must have the same dimension
##	#PIL.Image.fromarray(imageA).show()
##	#PIL.Image.fromarray(imageB).show()
##	err = ((imageA.astype("float") - imageB.astype("float")) ** 2).mean()
##	print 'error', err
##	# return the MSE, the lower the error, the more "similar"
##	# the two image
##	return err
##    for templ in template_candidates:
##        # determine similarity threshold
##        max_mse = ((thresh-templ)**2.0).mean() #255**2 #(templ**2.0).sum() / float((templ > 0).sum())
##        simil_thresh = max_mse * 0.15 # max 15% error of max error
##        print 'thresh',simil_thresh,max_mse
##        # look for similar template groups
##        simil = [(grouptempl,mse(templ,grouptempl)) for grouptempl in grouped_template_candidates]
##        simil = [(grouptempl,err) for grouptempl,err in simil if err < simil_thresh]
##        if simil:
##            # similar, add to group
##            mostsimil = sorted(simil, key=lambda(grouptempl,err): err)[0]
##            #grouped_template_candidates[mostsimil].append(templ)
##        else:
##            # not similar, create new group
##            #grouped_template_candidates[templ] = []
##            grouped_template_candidates.append(templ)
####    if debug:
####        #for templ,group in grouped_template_candidates.items():
####        for templ in grouped_template_candidates:
####            print templ.shape #, len(group)
####            PIL.Image.fromarray(templ).show()
##    template_candidates = grouped_template_candidates #.keys()
##    print 'grouped', len(template_candidates)
##
##    # search for templates
##    template_candidate_matches = []
##    for templ in template_candidates:
##        print 'searching for template matches', templ.shape
##        matches = []
##        for r in data:
##            x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##            x2 = x1+w
##            y2 = y1+h
##            fh = int(r['fontheight'])
##            
##            # extract subimg from buffer around text
##            buff = int(fh * 1)
##            buff_im_arr = im_arr[y1-buff:y2+buff, x1-buff:x2+buff].copy()
##            buff_im_arr[buff_im_arr == 255] = thresh
##            buff_im_arr = thresh - buff_im_arr # invert
##            buff_im_arr = buff_im_arr.astype(np.uint8)
##
##            # match template
##            templ = templ.astype(np.uint8)
##            res = cv2.matchTemplate(buff_im_arr, templ, cv2.TM_CCOEFF_NORMED)
##            thresh = 0.66
##            res[res < thresh] = 0
##
##            # get template centers
##            centers = np.nonzero(res)
##            centers = [(pt[1],pt[0]) for pt in np.transpose(centers)] # upperleft coords
##            centers = [(cx+templ.shape[1]/2.0, cy+templ.shape[0]/2.0) for cx,cy in centers] # move to center of template
##            centers = [(x1-buff+cx, y1-buff+cy) for cx,cy in centers] # offset to total img coords
##            matches.append(centers)
##
##        template_candidate_matches.append(matches)
##
##    # get the templates by number of matches
##    for templ,matches in sorted(zip(template_candidates, template_candidate_matches), key=lambda(t,ms): -len([centers for centers in ms if centers])):
##        # template
##        PIL.Image.fromarray(templ * 100).show()
##        # matches
##        if debug:
##            import pyagg
##            c = pyagg.canvas.from_image(PIL.Image.fromarray(im_arr))
##            print 'found', len([centers for centers in matches if centers])
##            for centers in matches:
##                for pt in centers:
##                    c.draw_circle(xy=pt, fillcolor=None, outlinecolor=(0,0,255), outlinewidth='1px')
##            c.get_image().show()
##
##    # assign
##    # ...
            
        


        

# OLD OLD
##def detect_toponym_anchors_distance(im, data, debug=False):
##    '''Detect anchor points via distance method, set each text's anchor point with the 'anchor' key.
##    - im must already be grayscale of possible anchor pixels.
##    - data is list of tesseract text dict results, preferably already filtered to toponyms. 
##    '''
##    #debug = True
####    lab = segmentation.rgb_to_lab(im)
####    l,a,b = lab.split()
####    
####    im_arr = 255 * (np.array(l) > (255/2)) # binarize on higher luminance values
####    im_arr = im_arr.astype(np.uint8)
####    #PIL.Image.fromarray(im_arr).show()
##
##    im_arr = np.array(im).astype(np.uint8)
##    im_arr_orig = im_arr.copy()
##    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
##    for r in data:
##        #if not r['placename']:
##        #    continue
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##
##        # possibly, narrow down the actual bbox via avg pixel coords
####        onxs,onys = np.nonzero(im_arr == 0)
####        tot = len(onxs)
####        onxcounts = zip(*onxs.unique(return_counts=True))
####        onycounts = zip(*onys.unique(return_counts=True))
####        # ...hmm...
##
##        buff = int(fh * 1)
##        filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff] = False # look in buffer zone around text box
##        filt_im_arr[y1:y2, x1:x2] = True # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##        #filt_im_arr[y1:y2, x1:x2] = True # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##    im_arr[filt_im_arr] = 255
##    #im_arr[im_arr < 255] = 0
##    if debug:
##        PIL.Image.fromarray(im_arr).show()
##
##    # determine kernel size from text height
##    #fh = sum([r['fontheight'] for r in data]) / len(data) # avg
##    fh = sorted([r['fontheight'] for r in data])[len(data)//2] # median
##    print 'font/kernel size', fh
##    print '(mean)',sum([r['fontheight'] for r in data]) / len(data) # avg
##
##    # get average value in neighbourhood
####    kernel = np.ones((h,h)) / (h*h)
####    im_arr = 255 - im_arr # invert
####    im_arr = cv2.filter2D(im_arr, -1, kernel) #cv2.blur(im_arr, (buff,buff)) #cv2.boxFilter(im_arr, -1, buff)
####    #im_arr = cv2.distanceTransform(im_arr, cv2.DIST_L1, 3)
####    print im_arr.max(), im_arr.mean()
####    PIL.Image.fromarray(im_arr).show()
##
##    # prep image to use in morphology
##    im_arr = 255 - im_arr # invert
##    ret,im_arr = cv2.threshold(im_arr,200,255,cv2.THRESH_BINARY)
##
##    # handle hollow shapes by filling them, using morphology closing
##    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(fh//8,fh//8))
##    #im_arr = cv2.morphologyEx(im_arr, cv2.MORPH_CLOSE, kernel)
##
##    # get distance to center
##    if debug:
##        PIL.Image.fromarray(im_arr).show()
##    im_arr = cv2.distanceTransform(im_arr, cv2.DIST_L2, 3)
##    dist_arr = im_arr.copy()
##    print 'dist max/mean', im_arr.max(), im_arr.mean()
##    if debug:
##        PIL.Image.fromarray(im_arr*50).show()
##
##    # only find actual anchors of significant size
##    # so limit to centers larger than 1/4th font height, but smaller than 1x font height
##    ch = fh/2.0 # value of center the size of fontsize will be half the fontsize
##    ch_lower = ch/4.0
##    ch_upper = ch
##    im_arr[im_arr < ch_lower] = 0
##    im_arr[im_arr > ch_upper] = 0
##    print 'limit to dist range', ch_lower, ch_upper
##
##    # get highest value in neighbourhood of size fontheight
##    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(fh,fh))
##    max_arr = cv2.dilate(im_arr,kernel,iterations=1)
##    if debug:
##        PIL.Image.fromarray(max_arr*50).show()
##
##    im_arr[im_arr != max_arr] = 0
##    #im_arr[im_arr < 100] = 0
##    #im_arr[im_arr == 0] = 0
##    #im_arr[(im_arr > 0) & (im_arr == max_arr)] = 255
##    print im_arr.max(), im_arr.mean()
##    #if debug:
##    #    PIL.Image.fromarray(im_arr*50).show()
##
##    # get shape centers
##    centers = np.nonzero(im_arr)
##    centervals = list(im_arr[centers]) #[im_arr[cx, cy] for cx,cy in centers]
##    centers = [(pt[1],pt[0]) for pt in np.transpose(centers)]
##    print 'centers', len(centers)
##
##    # link each text to closest point
##    if debug:
##        import pyagg
##        c = pyagg.canvas.from_image(PIL.Image.fromarray(im_arr_orig))
##        
##    points = []
##    for r in data:
##        #if not r['placename']:
##        #    continue
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        fh = int(r['fontheight'])
##        
##        if debug:
##            c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor=(0,255,0))
##        
##        # buffer
##        buff = int(fh * 1)
##        x1 -= buff
##        x2 += buff
##        y1 -= buff
##        y2 += buff
##        # first those within buffered bbox
##        nearby = filter(lambda(x,y): x1 < x < x2 and y1 < y < y2, centers)
##
##        if debug:
##            c.draw_box(bbox=[x1,y1,x2,y2], fillcolor=None, outlinecolor=(0,255,0))
##
##        if nearby:
##            # choose the nearest circle
####            nearest = sorted(nearby, key=lambda x: x[1])[0]
####            c = nearest[0]
####            print text,c
####            p = (int(c.centroid.x), int(c.centroid.y))
####            points.append((text, p))
##            
##            # or choose the highest value pixel (most concentrated)
##            maxpt = max(nearby, key=lambda pt: centervals[centers.index(pt)])
##            maxpt = map(int, maxpt)
##            r['anchor'] = maxpt
##
##            if debug:
##                for n in nearby:
##                    c.draw_circle(xy=n, fillsize=centervals[centers.index(n)], fillcolor=None, outlinecolor=(0,0,255))
##                c.draw_circle(xy=maxpt, fillsize=1, fillcolor=None, outlinecolor=(255,0,0))
##
##    if debug:
##        c.get_image().show()
##    
##    return data




        

