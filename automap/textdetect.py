
from . import segmentation

import numpy as np
import PIL, PIL.Image
from PIL import ImageOps

import re

import pytesseract



def run_ocr(im, bbox=None):
    if bbox:
        im = im.crop(bbox)
    data = pytesseract.image_to_data(im, lang='eng+fra', config='--psm 11') # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) for row in drows]
    
    # some standardizations
    for d in drows:
        for k in 'left top width height'.split():
            d[k] = int(d[k])
        d['conf'] = float(d['conf'])
        d['fontheight'] = d['height']
        
    return drows

def sniff_text_colors(im, samples=5, max_samples=8, max_texts=5):
    w,h = im.size
    bbox = [0,0,w,h]
    sw,sh = 200,200
    
    texts = []
    for i,q in enumerate(segmentation.sample_quads(bbox, (sw,sh))):
        print '# sample',i,q

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
                ls[ls < 0.66] = 0
                #PIL.Image.fromarray((ls*255).reshape((textim.size[1], textim.size[0]))).show()

                # get luminance weighted avg of colors
                r = np.average(rs, weights=ls)
                g = np.average(gs, weights=ls)
                b = np.average(bs, weights=ls)
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

                # calc max color diff in high luminance pixels
                textim = segmentation.quantize(textim)
                diff_arr = segmentation.color_difference(textim, textcol)
                coldiff = diff_arr.flatten()[ls > 0].max()
                print text['text'], textcol, diff_arr.flatten()[ls > 0].mean(), np.std(diff_arr.flatten()[ls > 0]), coldiff
                #diff_arr[ls.reshape(diff_arr.shape)==0] = 255.0
                #PIL.Image.fromarray(diff_arr).show()
                #textarr = np.array(textim)
                #textarr[diff_arr>coldiff] = (255,255,255)
                #PIL.Image.fromarray(textarr).show()
                
                #print textcol
                texts.append((text['text'],textcol,coldiff))
                
        if (i+1) >= 4: # minimum of 4 samples
            if len(texts) >= max_texts or (i+1) >= max_samples:
                break

    # group similar textcolors and return
    textcolors = [t[1] for t in texts]
    coldiffs = [t[2] for t in texts]
    colorgroups = segmentation.group_colors(textcolors, 10)
    print 'textcolors detected',[(col,len(cols)) for col,cols in colorgroups.items()]
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


def extract_texts(im, textcolors, threshold=25, textconf=60):
    '''
    - textcolors is list of colors.
    - threshold can be either single value used for all colors, or iterable of thresholds same length as textcolors.
    '''
    # extract from entire image
    w,h = im.size
    texts = []
    
    # upscale
    print 'upscaling'
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
        print 'isolating color', col, colthresh
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
        diff[diffmask] = 255 #colthresh
        lmask = diff

        # normalize
        #lmax,lmin = lmask.max(),lmask.min()
        #lmask = (lmask-lmin) / float(lmax-lmin) * 255.0
        #lmask[diffmask] = 255
        #print lmask.min(),lmask.max()

        lmaskim = PIL.Image.fromarray(lmask.astype(np.uint8))
        lmaskim.show()

        #imarr = np.array(upscale)
        #imarr[lmask==255] = (255,255,255)
        #PIL.Image.fromarray(imarr).show()
        
        # detect text
        print 'running ocr'
        data = run_ocr(lmaskim)
        print 'processing text'
        for text in data:
            
            # process text
            if text['conf'] > textconf: 
                
                # clean text
                text['text_clean'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end

                # ignore empty text
                if not text['text_clean']:
                    continue

                # enhance
                alphachars = [ch for ch in text['text_clean'] if ch.isalpha()]
                text['text_alphas'] = ''.join(alphachars)

                # maybe must have at least one alphanumeric to be considered text (ie just junk symbols)?
                # ... 
                
                # record info
                text['color'] = col
                textarr = lmask[text['top']:text['top']+text['height'], text['left']:text['left']+text['width']]
                text['color_match'] = textarr[textarr < colthresh].mean() # average diff of pixels below threshold

                # downscale coords
                for key in 'left top width height fontheight'.split():
                    text[key] = int( round(text[key] / 2.0) )

                # ignore tiny text (upscaling results in sometimes detecting ghost text from tiny pixel regions)
                if text['width'] <= 4 or text['height'] <= 4:
                    continue
                
                texts.append(text)

    return texts

def auto_detect_text(im, textcolors=None, colorthresh=25, textconf=60, sample=False, max_samples=8, max_texts=10, max_sniff_samples=8, max_sniff_texts=5):
    if not textcolors:
        print 'sniffing text colors'
        colorgroups = sniff_text_colors(im, max_samples=max_sniff_samples, max_texts=max_sniff_texts)
        # colors as color groupings
        textcolors = list(colorgroups.keys())
        # automatic detection of threshold for each textcolor
##        colorthresh = []
##        for col,colgroup in colorgroups.items():
##            gcols,gdiffs = zip(*colgroup)
##            # calc max diff from central group colors (diff required to incorporate all colors in the group)
##            pairdiffs = segmentation.color_differences([col] + list(gcols))
##            centraldiffs = [d for pair,d in pairdiffs.items() if col in pair]
##            diff = np.max(centraldiffs)
##            # add in the mean of each individual color diff
##            diff += np.mean(gdiffs)
##            colorthresh.append(diff)
        # debug
        segmentation.view_colors(colorgroups.keys())
        for group in colorgroups.values():
            groupcols = [subcol for subcol,coldiff in group]
            #segmentation.view_colors(groupcols)
    
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

    # sample text detection
    if sample:
        texts = sample_texts(im, textcolors, threshold=colorthresh, textconf=textconf, max_samples=max_samples, max_texts=max_texts)
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




