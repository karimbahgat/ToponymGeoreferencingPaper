
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
    sw,sh = 300,300
    texts = []
    for i,q in enumerate(segmentation.sample_quads(im, (sw,sh))):
        #print '---'
        print '# sample',i,q
        x1,y1,x2,y2 = q.bbox()
        sample = im.crop((x1,y1,x2,y2))
        lab = segmentation.rgb_to_lab(sample)
        l,a,b = lab.split()
        #print np.array(l).min(), np.array(l).max()
        lup = l.resize((l.size[0]*2,l.size[1]*2), PIL.Image.LANCZOS)
        #lup.show()
        data = run_ocr(lup)
        for text in data:
            #print '---',text
            if float(text['conf']) > 60 and len(text['text']) >= 5:
                # ignore nontoponyms
                if not text['text'].replace(' ',''):
                    # empty text
                    continue
                if not any((ch.isalpha() for ch in text['text'])):
                    # does not contain any alpha chars
                    continue
                if len([ch for ch in text['text'] if ch.isupper()]) > len(text['text']) / 2:
                    # more than half of characters is uppercase
                    continue
                # found text
                #print 'FOUND',text
                top,left = map(int, (text['top'],text['left']))
                width,height = map(int, (text['width'],text['height']))
                textbox = left/2.0,top/2.0,left/2.0+width/2.0,top/2.0+height/2.0
                textim = sample.crop(textbox)
                #textim.show()

                # get luminance weighted avg of colors
                rgbs = np.array(textim).reshape((textim.size[0]*textim.size[1],3))
                rs,gs,bs = rgbs[:,0],rgbs[:,1],rgbs[:,2]
                textlum = l.crop(textbox)
                #textlum.show()
                textlum = ImageOps.equalize(textlum)
                ls = np.array(textlum)
                ls = 1 - ((ls.flatten()-ls.min()) / float(ls.max()-ls.min()))
                ls[ls < 0.66] = 0
                #PIL.Image.fromarray((ls*255).reshape((textim.size[1], textim.size[0]))).show()
                r = np.average(rs, weights=ls)
                g = np.average(gs, weights=ls)
                b = np.average(bs, weights=ls)
                textcol = (r,g,b)
                #view_colors([textcol])
                
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
                
                #print textcol
                texts.append((text['text'],textcol))
        if i >= 3:
            if len(texts) >= max_texts or i >= max_samples:
                break
    textcolors = [t[1] for t in texts]
    textcolors = segmentation.group_colors(textcolors, 20)
    print 'textcolors detected',[(col,len(cols)) for col,cols in textcolors.items()]
    return textcolors

def sample_texts(im, textcolors, threshold=25, textconf=60, samplesize=(300,300), max_samples=8, max_texts=10):
    w,h = im.size
    sw,sh = samplesize
    texts = []
    # for each sample
##        for _ in range(samples):
##            print _
##            x,y = uniform(0,w-sw),uniform(0,h-sh)
    for i,q in enumerate(segmentation.sample_quads(im, (sw,sh))):
        #print '---'
        print '# sample',i,q
        x1,y1,x2,y2 = q.bbox()
        sample = im.crop((x1,y1,x2,y2))
        # upscale
        print 'upscaling'
        upscale = sample.resize((sample.size[0]*2,sample.size[1]*2), PIL.Image.LANCZOS)
        #lab = segmentation.rgb_to_lab(upscale)
        #l,a,b = lab.split()
        upscale = segmentation.quantize(upscale)
        #upscale.show()
        for col in textcolors:
            # calculate color difference
            print 'isolating color'
            diff = segmentation.color_difference(upscale, col)

            # mask based on color difference threshold
            diffmask = diff > threshold

            # maybe dilate to get edges?
##                from PIL import ImageMorph
##                diffmask = PIL.Image.fromarray(255-diffmask*255).convert('L')
##                op = ImageMorph.MorphOp(op_name='dilation8')
##                changes,diffmask = op.apply(diffmask)
##                diffmask = np.array(diffmask) == 0
##                # mask to luminance
##                lmask = np.array(l)
##                lmask[diffmask] = lmask.max() # cap max luminance to parts that are too different

            # OR mask to diff values
            diff[diffmask] = threshold
            lmask = diff

            # normalize
            lmax,lmin = lmask.max(),lmask.min()
            lmask = (lmask-lmin) / float(lmax-lmin) * 255.0
            #print lmask.min(),lmask.max()
            lmaskim = PIL.Image.fromarray(lmask.astype(np.uint8))
            #lmaskim.show()
            
            # detect text
            print 'running ocr'
            data = run_ocr(lmaskim)
            
            print 'processing text'
            for text in data:
                
                # process text
                if float(text['conf']) > textconf and len(text['text']) >= 2:
                    
                    # clean text
                    text['text_clean'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end

                    # ignore nontoponyms
                    if not text['text_clean'].replace(' ',''):
                        # empty text
                        continue
                    if not any((ch.isalpha() for ch in text['text_clean'])):
                        # does not contain any alpha chars
                        continue
                    if len([ch for ch in text['text_clean'] if ch.isupper()]) > len(text['text_clean']) / 2:
                        # more than half of characters is uppercase
                        continue

                    # record info
                    text['color'] = col

                    # downscale coords
                    for key in 'left top width height'.split():
                        text[key] = int( round(text[key] / 2.0) )

                    # ignore tiny text
                    if text['width'] <= 4 or text['height'] <= 4:
                        continue

                    # ignore text along edges (could be cutoff)
                    edgebuff = text['height']
                    if text['left'] < edgebuff or text['top'] < edgebuff \
                       or (text['left']+text['width']) > sw-edgebuff or (text['top']+text['height']) > sh-edgebuff:
                        #print 'edge case',text
                        #print [edgebuff,edgebuff,sw-edgebuff,sh-edgebuff]
                        continue

                    # convert sample space to image space
                    text['left'] = int(x1 + text['left'])
                    text['top'] = int(y1 + text['top'])
                    texts.append(text)

        print 'texts',len(texts)
        if i >= 3:
            if len(texts) >= max_texts or i >= max_samples:
                break
                
    return texts


def extract_texts(im, textcolors, threshold=25, textconf=60):
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
    
    for col in textcolors:
        # calculate color difference
        print 'isolating color', col
        diff = segmentation.color_difference(upscale, col)

        # mask based on color difference threshold
        diffmask = diff > threshold

        # maybe dilate to get edges?
##                from PIL import ImageMorph
##                diffmask = PIL.Image.fromarray(255-diffmask*255).convert('L')
##                op = ImageMorph.MorphOp(op_name='dilation8')
##                changes,diffmask = op.apply(diffmask)
##                diffmask = np.array(diffmask) == 0
##                # mask to luminance
##                lmask = np.array(l)
##                lmask[diffmask] = lmask.max() # cap max luminance to parts that are too different

        # OR mask to diff values
        diff[diffmask] = 255 # = threshold
        lmask = diff

        # normalize
##        lmax,lmin = lmask.max(),lmask.min()
##        lmask = (lmask-lmin) / float(lmax-lmin) * 255.0
##        #print lmask.min(),lmask.max()

        lmaskim = PIL.Image.fromarray(lmask.astype(np.uint8))
        #lmaskim.show()
        
        # detect text
        print 'running ocr'
        data = run_ocr(lmaskim)
        print 'processing text'
        for text in data:
            
            # process text
            if float(text['conf']) > textconf and len(text['text']) >= 2:
                
                # clean text
                text['text_clean'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end

                # ignore nontoponyms
                if not text['text_clean'].replace(' ',''):
                    # empty text
                    continue
                if not any((ch.isalpha() for ch in text['text_clean'])):
                    # does not contain any alpha chars
                    continue
                if len([ch for ch in text['text_clean'] if ch.isupper()]) > len(text['text_clean']) / 2:
                    # more than half of characters is uppercase
                    continue
                
                # record info
                text['color'] = col

                # downscale coords
                for key in 'left top width height'.split():
                    text[key] = int( round(text[key] / 2.0) )

                # ignore tiny text
                if text['width'] <= 4 or text['height'] <= 4:
                    continue
                
                texts.append(text)

    return texts

def auto_detect_text(im, textcolors=None, colorthresh=25, textconf=60, sample=False, max_samples=8, max_texts=10, max_sniff_samples=8, max_sniff_texts=5):
    if not textcolors:
        print 'sniffing text colors'
        textcolors = sniff_text_colors(im, max_samples=max_sniff_samples, max_texts=max_sniff_texts)
        #segmentation.view_colors(textcolors)
    
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




