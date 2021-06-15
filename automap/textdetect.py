
from . import segmentation

import numpy as np
import PIL, PIL.Image
from PIL import ImageOps

import re

import pytesseract

try:
    basestring
except:
    basestring = (bytes,str)


#pytesseract.pytesseract.tesseract_cmd = '/home/cdsw/conda/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'


def run_ocr(im, bbox=None, mode=11):
    if bbox:
        xoff,yoff = bbox[:2]
        im = im.crop(bbox)
    data = pytesseract.image_to_data(im, lang='eng+fra', config='--psm {}'.format(mode)) # +equ
    #data = pytesseract.image_to_data(im, lang='eng+fra', config='--psm {} --tessdata-dir "{}"'.format(mode, r'C:\Users\kimok\Desktop\tessdata_fast')) # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) 
            for row in drows
            if len(row) == len(dfields)]
    
    # some standardizations
    for d in drows:
        for k in 'left top width height'.split():
            d[k] = int(d[k])
        if bbox:
            d['left'] += xoff
            d['top'] += yoff
        d['conf'] = float(d['conf'])
        d['fontheight'] = d['height']
        
    return drows

def refine_textbox(im_arr, textdata):
    x,y = textdata['left'],textdata['top']
    w,h = textdata['width'],textdata['height']

    debug = False
##    debug = ('BENIN' in textdata['text'] or \
##             'BAM' in textdata['text'] or \
##             'OUDALAN' in textdata['text'] or \
##             'GHANA' in textdata['text'] or \
##             'Wa' in textdata['text'] or \
##             'ound' in textdata['text'] or \
##             'Toug' in textdata['text'] or \
##             'Krach' in textdata['text'] or \
##             'aux' in textdata['text'] or \
##             'cass' in textdata['text'] or \
##             'Kaya' in textdata['text'] or \
##             'Nouna' in textdata['text'] or \
##             'Navronge' in textdata['text'])
    histdebug = False
    changedebug = True

    # textbox histograms is not very accurate for short text strings, skip
    if len(textdata['text']) < 3:
        return textdata

    # crop img to box
    im_box = im_arr[y:y+h, x:x+w]
    if im_box.shape[1] <= 4 or im_box.shape[0] <= 4:
        return textdata # too small
    if debug and histdebug:
        PIL.Image.fromarray(255-im_box).show()

    # calc y hist
    horiz = []
    ys = list(range(y, y+h))
    for boxy in range(im_box.shape[0]):
        summ = (im_box[boxy,:] < 255).sum()
        horiz.append(summ)

    # define upper and lower font core as first and last time hist reaches more than 90% of above-avg mean
    histmean = np.mean(horiz)
    histmeanpos = np.mean([v for v in horiz if v > histmean])
    histthresh = histmeanpos * 0.9
    ystart = textdata['top']
    for y,ycount in zip(ys,horiz):
        if ycount >= histthresh:
            ystart = y
            break
    yend = textdata['top'] + textdata['height']
    for y,ycount in reversed(list(zip(ys,horiz))):
        if ycount >= histthresh:
            yend = y
            break

    # view
    if debug and histdebug:
        import matplotlib.pyplot as plt
        plt.gca().invert_yaxis()
        plt.barh(ys, horiz)
        plt.axvline(histthresh, color='blue')
        plt.axhline(ystart, color='black')
        plt.axhline(yend, color='black')
        plt.show()

    # calc x hist
    vertic = []
    xs = list(range(x, x+w))
    for x in range(im_box.shape[1]):
        summ = (im_box[:,x] < 255).sum()
        vertic.append(summ)

    # define left and right text boundary as first and last time hist reaches more than 90% of above-avg mean
    histmean = np.mean(vertic)
    histmeanpos = np.mean([v for v in vertic if v > histmean])
    histthresh = histmeanpos * 0.9
    xstart = textdata['left']
    for x,xcount in zip(xs,vertic):
        if xcount >= histthresh:
            xstart = x
            break
    xend = textdata['left'] + textdata['width']
    for x,xcount in reversed(list(zip(xs,vertic))):
        if xcount >= histthresh:
            xend = x
            break

    # view
    if debug and histdebug:
        import matplotlib.pyplot as plt
        plt.bar(xs, vertic)
        plt.axhline(histthresh, color='blue')
        plt.axvline(xstart, color='black')
        plt.axvline(xend, color='black')
        plt.show()

    # calc bbox change as percent of font height
    x1change = (xstart-textdata['left']) / float(textdata['height'])
    x2change = (xend-(textdata['left']+textdata['width'])) / float(textdata['height'])
    y1change = (ystart-textdata['top']) / float(textdata['height'])
    y2change = (yend-(textdata['top']+textdata['height'])) / float(textdata['height'])
    if debug:
        print('text', textdata['text'], (textdata['width'],textdata['height']), 'was at x1,x2,y1,y2', textdata['left'], textdata['left']+textdata['width'], textdata['top'], textdata['top']+textdata['height'])
        print('refined to x1,x2,y1,y2', xstart, xend, ystart, yend)
        print('as percent', x1change, x2change, y1change, y2change)

    # only change boundaries that change beyond threshold
    xchangethresh = 1.5 # left right moved by >1.5x font height
    ychangethresh = 0.5 # top bottom moved by >50% font height
    h = yend-ystart
    if x1change > xchangethresh:
        xstart -= int(round(h)) # expand the left by 1x the font core
    else:
        xstart = textdata['left'] # set back to original
        
    if x2change < -xchangethresh:
        xend += int(round(h)) # expand the right edge by 1x the font core
    else:
        xend = textdata['left']+textdata['width'] # set back to original
        
    if y1change > ychangethresh:
        ystart -= int(round(h/2.0)) # expand the font core vertically to upper fourth (font core is two fourths)
    else:
        ystart = textdata['top'] # set back to original
        
    if y2change < -ychangethresh:
        yend += int(round(h/2.0)) # expand the font core vertically to lower fourth (font core is two fourths)
    else:
        yend = textdata['top']+textdata['height'] # set back to original

    # modify the bbox
    if x1change > xchangethresh or x2change < -xchangethresh or y1change > ychangethresh or y2change < -ychangethresh:
        # debug before
        if debug and changedebug:
            PIL.Image.fromarray(255-im_box).show()

        # limit to within img
        ystart,yend = max(0,ystart), min(im_arr.shape[0], yend)
        xstart,xend = max(0,xstart), min(im_arr.shape[1], xend)

        # update textbox
        textdata['left'] = xstart
        textdata['width'] = xend-xstart
        textdata['top'] = ystart
        textdata['height'] = yend-ystart
        textdata['fontheight'] = textdata['height']

        # crop img to refined box
        im_box = im_arr[ystart:yend, xstart:xend]
        if im_box.shape[1] <= 4 or im_box.shape[0] <= 4:
            return textdata # too small

        # debug after
        if debug and changedebug:
            PIL.Image.fromarray((255-im_box).astype(np.uint8)).show()

##        # rerun single line ocr
##        # psm 7 = Treat the image as a single text line.
##        # psm 8 = Treat the image as a single word.
##        # psm 13 is used with the new LSTM engine to OCR a single textline image.
##        newim = PIL.Image.fromarray(im_box.astype(np.uint8))
##        origtextdata = textdata
##        result = run_ocr(newim, mode=13) 
##        textdata = sorted(result, key=lambda d: d['conf'])[-1] # for some reason returns multiple junk results, only use the highest confidence result
##        textdata['left'] += xstart # offset relative to whole img
##        textdata['top'] += ystart # offset relative to whole img
##        textdata['text'] = textdata.pop('text', '') # in case no text
##        
##        fromdims = [origtextdata[k] for k in 'left top width height'.split()]
##        todims = [textdata[k] for k in 'left top width height'.split()]
##        print u'ocr rerun from "{}" ({}) to "{}" ({})'.format(origtextdata['text'], fromdims, textdata['text'], todims)

    return textdata

def sniff_text_colors(im, seginfo=None, min_samples=4, max_samples=4+4**2, max_texts=3):
    w,h = im.size

    xmin,ymin,xmax,ymax = 0,0,w,h
    if seginfo:
        mapregion = next((f['geometry'] for f in seginfo['features'] if f['properties']['type'] == 'Map'), None)
        if mapregion:
            xs,ys = zip(*[p for p in mapregion['coordinates'][0]])
            xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs),max(ys)
    bbox = [xmin,ymin,xmax,ymax]
    print('sniffing inside', bbox)

    sw,sh = 300,300
    
    texts = []
    for i,q in enumerate(segmentation.sample_quads(bbox, (sw,sh))):
        print('# sample',i,q)

        # crop sample of image
        x1,y1,x2,y2 = q.bbox()
        sample = im.crop((x1,y1,x2,y2))

        # convert to luminance
        lab = segmentation.rgb_to_lab(sample)
        l,a,b = lab.split()

        # upscale and run ocr
        lup = l.resize((l.size[0]*2,l.size[1]*2), PIL.Image.LANCZOS)
        #lup.show()
        data = run_ocr(lup)

        # loop detected texts
        for text in data:
            #print '---',text

            # samples must be good examples
            if text['conf'] > 60 and len(text['text']) >= 4:
                
                # found text, crop img
                #print 'SNIFFING text',text
                top,left = text['top'],text['left']
                width,height = text['width'],text['height']
                textbox = left/2.0,top/2.0,left/2.0+width/2.0,top/2.0+height/2.0
                textim = sample.crop(textbox)
                #textim.show()

                # get color values and luminance
                rgbs = np.array(textim).reshape((textim.size[0]*textim.size[1],3))
                rs,gs,bs = rgbs[:,0],rgbs[:,1],rgbs[:,2]
                textlum = l.crop(textbox)
                #textlum.show()

                # equalize luminance and ignore the bottom 66% luminant pixels
                textlum = ImageOps.equalize(textlum)
                ls = np.array(textlum)
                ls = 1 - ((ls.flatten()-ls.min()) / float(ls.max()-ls.min()))
                ls_top = ls.copy()
                ls_top[ls < 0.66] = 0
                #PIL.Image.fromarray((ls_top*255).reshape((textim.size[1], textim.size[0]))).show()

                # get luminance weighted avg of colors
                r = np.average(rs, weights=ls_top)
                g = np.average(gs, weights=ls_top)
                b = np.average(bs, weights=ls_top)
                textcol = (r,g,b)
                #segmentation.view_colors([textcol])
                
                # avg smoothed midline approach
##                    foreground = textim.filter(ImageFilter.MinFilter(3))
##                    avg = foreground.filter(ImageFilter.BoxBlur(7))
##                    #avg.show()
##                    #hist = avg.getcolors(textim.size[0]*textim.size[1])
##                    midline = np.array(avg)[avg.size[1]/2,:,:]
##                    cols,counts = np.unique(midline, axis=0, return_counts=True)
##                    cols = map(tuple, cols)
##                    hist = zip(counts,cols)
##                    
##                    cols = [rgb for c,rgb in hist]
##                    cols = group_colors(cols)
##                    #view_colors(cols.keys())
##                    cols = sorted(cols.items(), key=lambda(k,v): -len(v))
##                    #print cols
##                    textcol = cols[0][0]

                #foreground = textim.filter(ImageFilter.MinFilter(5)) # darkest
                #foreground.show()
                #hist = foreground.getcolors(textim.size[0]*textim.size[1])

                #midline = np.array(textim)[textim.size[1]/2,:,:]
                #cols,counts = np.unique(midline, axis=0, return_counts=True)
                #cols = map(tuple, cols)
                #hist = zip(counts,cols)
                
                #hist = sorted(hist, key=lambda(c,rgb): -c)
                #textcol = hist[0][1]

                # get luminance weighted avg of color diff from detected color
                # (TODO: maybe should be mean+std?)
                # (TODO: switch so only based on the same pixels that made up the color detection, ie weighted avg of top 66% of luminance pixels)
                textim = segmentation.quantize(textim)
                diff_arr = segmentation.color_difference(textim, textcol)
                ls_top = ls.copy()
                ls_top[ls < 0.33] = 0 # lower, to also include distance required to capture edge pixels
                #PIL.Image.fromarray((ls_top*255).reshape((textim.size[1], textim.size[0]))).show()

                coldiff = np.average(diff_arr.flatten(), weights=ls_top)
                
                #maskdiffs = diff_arr.flatten()[ls > 0.33]
                #print text['text'], textcol, maskdiffs.mean(), np.std(maskdiffs), maskdiffs.max()
                #coldiff = maskdiffs.max()
                
                #diff_arr[ls.reshape(diff_arr.shape)==0] = 255.0
                #PIL.Image.fromarray(diff_arr).show()
                #textarr = np.array(textim)
                #textarr[diff_arr>coldiff] = (255,255,255)
                #PIL.Image.fromarray(textarr).show()
                
                #print textcol
                texts.append((text['text'],textcol,coldiff))
                
        if (i+1) >= min_samples: # minimum quad samples
            if len(texts) >= max_texts or (i+1) >= max_samples:
                break

    # group similar textcolors and return
    textcolors = [t[1] for t in texts]
    coldiffs = [t[2] for t in texts]
    colorgroups = segmentation.group_colors(textcolors, 15)
    print('textcolors detected',[(col,len(cols)) for col,cols in colorgroups.items()])
    #segmentation.view_colors(textcolors)
    
    # pair each group color member with their coldiff
    for col in list(colorgroups.keys()):
        colorgroups[col] = [(subcol,coldiffs[textcolors.index(subcol)]) for subcol in colorgroups[col]]
    return colorgroups

##def sample_texts(im, textcolors, threshold=25, textconf=60, samplesize=(300,300), max_samples=8, max_texts=10):
##    # REMEMBER: update text processing/filtering to be same as extract_texts(), maybe by calling it?
##    # ...
##    raise Exception('Sample extraction of texts not finished, must be updated to be same as extract_texts()')
##
##    w,h = im.size
##    sw,sh = samplesize
##    texts = []
##    
##    # for each sample
####        for _ in range(samples):
####            print _
####            x,y = uniform(0,w-sw),uniform(0,h-sh)
##    
##    for i,q in enumerate(segmentation.sample_quads(im, (sw,sh))):
##        #print '---'
##        print '# sample',i,q
##        x1,y1,x2,y2 = q.bbox()
##        sample = im.crop((x1,y1,x2,y2))
##        # upscale
##        print 'upscaling'
##        upscale = sample.resize((sample.size[0]*2,sample.size[1]*2), PIL.Image.LANCZOS)
##        #lab = segmentation.rgb_to_lab(upscale)
##        #l,a,b = lab.split()
##        upscale = segmentation.quantize(upscale)
##        #upscale.show()
##        for col in textcolors:
##            # calculate color difference
##            print 'isolating color'
##            diff = segmentation.color_difference(upscale, col)
##
##            # mask based on color difference threshold
##            diffmask = diff > threshold
##
##            # maybe dilate to get edges?
####                from PIL import ImageMorph
####                diffmask = PIL.Image.fromarray(255-diffmask*255).convert('L')
####                op = ImageMorph.MorphOp(op_name='dilation8')
####                changes,diffmask = op.apply(diffmask)
####                diffmask = np.array(diffmask) == 0
####                # mask to luminance
####                lmask = np.array(l)
####                lmask[diffmask] = lmask.max() # cap max luminance to parts that are too different
##
##            # OR mask to diff values
##            diff[diffmask] = threshold
##            lmask = diff
##
##            # normalize
##            lmax,lmin = lmask.max(),lmask.min()
##            lmask = (lmask-lmin) / float(lmax-lmin) * 255.0
##            #print lmask.min(),lmask.max()
##            lmaskim = PIL.Image.fromarray(lmask.astype(np.uint8))
##            #lmaskim.show()
##            
##            # detect text
##            print 'running ocr'
##            data = run_ocr(lmaskim)
##            
##            print 'processing text'
##            for text in data:
##                
##                # process text
##                if float(text['conf']) > textconf and len(text['text']) >= 2:
##                    
##                    # clean text
##                    text['text_clean'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end
##
##                    # ignore nontoponyms
##                    if not text['text_clean'].replace(' ',''):
##                        # empty text
##                        continue
##                    if not any((ch.isalpha() for ch in text['text_clean'])):
##                        # does not contain any alpha chars
##                        continue
##                    if len([ch for ch in text['text_clean'] if ch.isupper()]) > len(text['text_clean']) / 2:
##                        # more than half of characters is uppercase
##                        continue
##
##                    # record info
##                    text['color'] = col
##
##                    # downscale coords
##                    for key in 'left top width height'.split():
##                        text[key] = int( round(text[key] / 2.0) )
##
##                    # ignore tiny text
##                    if text['width'] <= 4 or text['height'] <= 4:
##                        continue
##
##                    # ignore text along edges (could be cutoff)
##                    edgebuff = text['height']
##                    if text['left'] < edgebuff or text['top'] < edgebuff \
##                       or (text['left']+text['width']) > sw-edgebuff or (text['top']+text['height']) > sh-edgebuff:
##                        #print 'edge case',text
##                        #print [edgebuff,edgebuff,sw-edgebuff,sh-edgebuff]
##                        continue
##
##                    # convert sample space to image space
##                    text['left'] = int(x1 + text['left'])
##                    text['top'] = int(y1 + text['top'])
##                    texts.append(text)
##
##        print 'texts',len(texts)
##        if i >= 3:
##            if len(texts) >= max_texts or i >= max_samples:
##                break
##                
##    return texts


def extract_texts_parallel(im, textcolors, threshold=25, textconf=60, max_procs=None, tilesize=(500,500)):
    w,h = im.size
    tw,th = map(int, tilesize)
    texts = []
    tile_overlap = 0.2 # one fifth

    # div image into bboxes
##    boxes = []
##    for y1 in range(0, h+1, th):
##        for x1 in range(0, w+1, tw):
##            x2,y2 = min(x1+tw, w), min(y1+th, h)
##            box = [x1,y1,x2,y2]
##            boxes.append(box)
    
    # div image into bboxes
    boxes = []
    for y1 in range(0, h+1, th-int(th*tile_overlap)):
        y2 = y1+th
        y2 = min(y2, h) # cap at img limits
        #y1 = y2 - th # enforce tile size (extend back in case of small ending sliver)
        for x1 in range(0, w+1, tw-int(tw*tile_overlap)):
            x2 = x1+tw
            x2 = min(x2, w) # cap at img limits
            #x1 = x2 - tw # enforce tile size (extend back in case of small ending sliver)
            box = [x1,y1,x2,y2]
            boxes.append(box)

    # save im to temp path, so processes can access
    import tempfile
    temppath = tempfile.mktemp() + '.png'
    im.save(temppath)

    # loop bboxes, starting new subprocess of extract_texts
    import multiprocessing as mp
    max_procs = max_procs or mp.cpu_count() - 1 # excluding main process

    # pool
##    print 'procs',max_procs
##    pool = mp.Pool(max_procs) #, maxtasksperchild=1)
##    kwargs = [dict(im=temppath,
##                   textcolors=textcolors,
##                   threshold=threshold,
##                   textconf=textconf,
##                   bbox=box,
##                   )
##              for box in boxes]
##    texts = []
##    def get_results(ptexts):
##        print 'finished',len(ptexts)
##        texts.extend(ptexts)
##    for dct in kwargs:
##        pool.apply_async(extract_texts, None, dct, callback=get_results)
##    for w in pool._pool:
##        print repr(w.is_alive())
##    print 'waiting'
##    pool.close()
##    pool.join()
##    for w in pool._pool:
##        print repr(w.is_alive())
##    print 'results',len(texts)

    # semimanual
    if not isinstance(textcolors, list):
        textcolors = [textcolors]
    if not isinstance(threshold, list):
        threshold = [threshold for _ in textcolors]

    with mp.Pool(max_procs) as pool:
        procs = []
        results = []
        texts = []

        # initiate processes stepwise
        for i,box in enumerate(boxes):
            print('processing img tile', box, i+1, 'of', len(boxes))
            
            for textcolor,thresh in zip(textcolors, threshold):
                print('color',textcolor)
                
                # manual procs
                p = pool.apply_async(extract_texts,
                                     kwds=dict(im=temppath,
                                               textcolors=[textcolor],
                                               threshold=thresh,
                                               textconf=textconf,
                                               bbox=box,
                                               ),
                                     )
                procs.append(p)
                results.append((p,box))

                # wait in line
                while len(procs) >= max_procs:
                    for p in procs:
                        if p.ready():
                            procs.remove(p)

        # get results of all processes
        for p,box in results:
            try:
                boxtexts = []
                # ignore any text closer than 1x fontheight away from box edges
                restexts = p.get(timeout=None)
                #print 'texts for', box
                #print 'orig',len(restexts)
                for text in restexts:
                    if text['left'] < (box[0] + text['fontheight']):
                        continue
                    if text['top'] < (box[1] + text['fontheight']):
                        continue
                    if (text['left']+text['width']) > (box[2] - text['fontheight']):
                        continue
                    if (text['top']+text['height']) > (box[3] - text['fontheight']):
                        continue
                    boxtexts.append(text)
                #print 'after dropping edge texts',len(boxtexts)
                texts.extend(boxtexts)
            except ValueError:
                # weird error if empty results (i think)
                pass

    # manual
##    procs = []
##    texts = []
##    for i,box in enumerate(boxes):
##        print box, i+1, 'of', len(boxes)
##
##        # normal debug
####        ptexts = extract_texts(im=temppath,
####                               textcolors=textcolors,
####                               threshold=threshold,
####                               textconf=textconf,
####                               bbox=box,
####                               )
####        texts.extend(ptexts)
####        continue
##
##        # manual procs
##        p = mp.Process(target=extract_texts,
##                       kwargs=dict(im=temppath,
##                                   textcolors=textcolors,
##                                   threshold=threshold,
##                                   textconf=textconf,
##                                   bbox=box,
##                                   ),
##                       )
##        p.daemon = True
##        p.start()
##        procs.append(p)
##
##        # Wait in line
##        while len(procs) >= max_procs:
##            for p in procs:
##                if not p.is_alive():
##                    procs.remove(p)
##
##    # final check that remaining procs have finished
##    for p in procs:
##        p.join(None)

    return texts


def extract_texts(im, textcolors, threshold=25, textconf=60, bbox=None):
    '''
    - textcolors is list of colors.
    - threshold can be either single value used for all colors, or iterable of thresholds same length as textcolors.
    '''
    # load from file if string
    if isinstance(im, basestring):
        im = PIL.Image.open(im)

    # crop to bbox if specified
    if bbox:
        im = im.crop(tuple(bbox))
    
    # extract from entire image
    w,h = im.size
    texts = []
    
    # upscale
    print('upscaling')
    upscale = im.resize((im.size[0]*2,im.size[1]*2), PIL.Image.LANCZOS)
    #lab = segmentation.rgb_to_lab(upscale)
    #l,a,b = lab.split()
    upscale = segmentation.quantize(upscale)
    #upscale.show()

    if isinstance(threshold, (int,float)):
        threshold = [threshold for col in textcolors]

    assert len(textcolors) == len(threshold)
    
    for col,colthresh in zip(textcolors,threshold):
        # calculate color difference
        print('isolating color', col, colthresh)
        diff = segmentation.color_difference(upscale, col)

        # mask based on color difference threshold
        diffmask = diff > colthresh

        # maybe dilate to get edges?
##        from PIL import ImageMorph
##        diffmask = PIL.Image.fromarray(255-diffmask*255).convert('L')
##        op = ImageMorph.MorphOp(op_name='dilation8')
##        changes,diffmask = op.apply(diffmask)
##        diffmask = np.array(diffmask) == 0

        # mask to luminance
        #lmask = np.array(l)
        #lmask[diffmask] = lmask.max() # cap max luminance to parts that are too different

        # OR mask to diff values
        diff[diffmask] = colthresh
        lmask = diff

        # normalize
        lmax,lmin = lmask.max(),lmask.min()
        lmask = (lmask-lmin) / float(lmax-lmin) * 127.0 # lowest matching color is half gray
        lmask[diffmask] = 255
        #print lmask.min(),lmask.max()

        lmaskim = PIL.Image.fromarray(lmask.astype(np.uint8))
        #lmaskim.show()

        #imarr = np.array(upscale)
        #imarr[lmask==255] = (255,255,255)
        #PIL.Image.fromarray(imarr).show()
        
        # detect text
        print('running ocr')
        data = run_ocr(lmaskim)
        print('processing text')
        for text in data:
            
            # process text
            if text['conf'] > textconf:

                # refine ocr
                #print 'text orig -->', text
                text = refine_textbox(lmask, text)
                #print 'text refined -->', text
                
                # clean text
                text['text_clean'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end

                # ignore empty text
                if not text['text_clean']:
                    continue

                # enhance
                alphachars = [ch for ch in text['text_clean'] if ch.isalpha()]
                text['text_alphas'] = ''.join(alphachars)

                # must have at least three alphanumeric to be considered text (otherwise likely just junk symbols)?
                if len(text['text_alphas']) <= 2:
                    continue
                
                # record info
                text['color'] = col
                textdiffarr = diff[text['top']:text['top']+text['height'], text['left']:text['left']+text['width']]
                text['color_match'] = float(textdiffarr[textdiffarr < colthresh].mean()) # average diff of pixels below threshold

                # downscale coords
                for key in 'left top width height fontheight'.split():
                    text[key] = int( round(text[key] / 2.0) )

                # ignore tiny text (upscaling results in sometimes detecting ghost text from tiny pixel regions)
                if text['width'] <= 4 or text['height'] <= 4:
                    continue

                # offset coords if bbox
                if bbox:
                    xoff,yoff = bbox[:2]
                    text['left'] += xoff
                    text['top'] += yoff
                
                texts.append(text)

    return texts

def auto_detect_text(im, textcolors=None, colorthresh=25, textconf=60, parallel=False, sample=False, seginfo=None, max_procs=None, max_samples=8, max_texts=10, max_sniff_samples=4+4**2, max_sniff_texts=3):
    if not textcolors:
        print('sniffing text colors')
        colorgroups = sniff_text_colors(im, seginfo=seginfo, max_samples=max_sniff_samples, max_texts=max_sniff_texts)

        # colors as color groupings
        textcolors = list(colorgroups.keys())
        
        # automatic detection of threshold for each textcolor
        colorthresh = []
        for col,colgroup in colorgroups.items():
            gcols,gdiffs = zip(*colgroup)
            # calc mean diff from central group colors
            pairdiffs = segmentation.color_differences([col] + list(gcols))
            centraldiffs = [d for pair,d in pairdiffs.items() if col in pair]
            centraldiff = np.mean(centraldiffs)
            #print 'color group differences',col,centraldiff,centraldiffs
            diff = centraldiff
            # add in the mean of each individual color diff
            gdiff = np.mean(gdiffs)
            #print 'color internal pixel differences',col,gdiff,gdiffs
            diff += gdiff
            # cap at 30
            diff = min(diff, 30)
            colorthresh.append(diff)
            
        # debug
        #segmentation.view_colors(colorgroups.keys())
##        for group in colorgroups.values():
##            groupcols = [subcol for subcol,coldiff in group]
##            segmentation.view_colors(groupcols)

        # fallback on black if no samples detected
        if not textcolors:
            textcolors = [(0,0,0)]
            colorthresh = [25]
    
    # compare with just luminance
##    lab = rgb_to_lab(im)
##    l,a,b = lab.split()
##    l.show()
##    im = l
##    textcolors = [(0,0,0)]

    # whole img ocr comparison
##    print 'upscaling'
##    upscale = im.resize((im.size[0]*2,im.size[1]*2), PIL.Image.LANCZOS)
##    upscale = quantize(upscale)
##    #upscale.show()
##    texts = []
##    for col in textcolors:
##        # isolate color
##        print 'isolating color'
##        diff = color_difference(upscale, (0,0,0))
##        #diff.show()
##        # detect text
##        print 'running ocr'
##        d = automap.main.detect_data(diff)
##        texts.extend(d)
##
##    print time()-t

    # run text detection
    if sample:
        texts = sample_texts(im, textcolors, threshold=colorthresh, textconf=textconf, max_samples=max_samples, max_texts=max_texts)
    elif parallel:
        texts = extract_texts_parallel(im, textcolors, threshold=colorthresh, textconf=textconf, max_procs=max_procs)
    else:
        texts = extract_texts(im, textcolors, threshold=colorthresh, textconf=textconf)
    
##    for t in texts:
##        print t
        
##    import pyagg
##    c = pyagg.canvas.from_image(im)
##    c.pixel_space()
##    for t in texts:
##        left,top,width,height = [t[k] for k in 'left top width height'.split()]
##        c.draw_box(bbox=[left,top,left+width,top+height], fillcolor=None, outlinecolor=(0,255,0), outlinewidth='2px')
##        c.draw_text(t['text'], xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0))
##    c.get_image().show()

    return texts




