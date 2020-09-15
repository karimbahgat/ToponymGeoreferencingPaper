
import automap
from automap.main import quantize, mask_text, image_segments, threshold, detect_data, process_text, detect_text_points, connect_text, find_matches, warp
from automap.main import maincolors, color_edges
from automap.rmse import polynomial, predict

import PIL, PIL.Image

import sqlite3
import json
import os
import warnings
import itertools

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff_matrix import delta_e_cie2000

import numpy as np
import cv2

warnings.simplefilter('always')

def smooth(im):
    im_arr = np.array(im)
    im_arr = cv2.bilateralFilter(im_arr,9,75,75)
    im = PIL.Image.fromarray(im_arr)
    return im

def color_difference_ims(im, im2):
    # very slow....
    diffs = np.zeros((im.size[1],im.size[0]))
    imar = np.array(im)
    imar2 = np.array(im2)
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            c = imar[y,x]
            c2 = imar2[y,x]
            d = delta_e_cie2000(convert_color(sRGBColor(*c, is_upscaled=True), LabColor).get_value_tuple(),
                                np.array([convert_color(sRGBColor(*c2, is_upscaled=True), LabColor).get_value_tuple()]),
                                )
            diffs[y,x] = d
    return diffs

def color_difference(im, color, thresh=25):
    im_arr = np.array(im)

    #target = color
    #diffs = [pairdiffs[tuple(sorted([target,oth]))] for oth in colors if oth != target]

    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
    counts,colors = zip(*im.getcolors(256))
    #colors,counts = np.unique(im_arr, return_counts=True)
    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
                  for col in colors]
    diffs = delta_e_cie2000(target, np.array(colors_lab))
    
    difftable = dict(list(zip(colors,diffs)))
    diff_im = np.zeros((im.height,im.width))
    for col,diff in difftable.items():
        #print col
        diff_im[(im_arr[:,:,0]==col[0])&(im_arr[:,:,1]==col[1])&(im_arr[:,:,2]==col[2])] = diff # 3 lookup is slow

    dissim = diff_im > thresh
    diff_im[dissim] = 255
    diff_im = (diff_im - diff_im.min()) / diff_im.max() * 255
    im_arr = diff_im.astype(np.uint8)

##    maxdiff = diff_im.max()
##    diff_im = diff_im / maxdiff * 255
##    im_arr = diff_im.astype(np.uint8)

    # TODO: Maybe also do gaussian or otsu binarization/smoothing?
    # Seems to do worse than original, makes sense since loses/changes original information
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    #import cv2
    #im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
    #im_arr = cv2.GaussianBlur(im_arr,(3,3),0)
    #ret,im_arr = cv2.threshold(im_arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    im = PIL.Image.fromarray(im_arr)
    return im
                     
def color_differences(colors):
    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
                  for col in colors]
    pairdiffs = dict()
    for lcol1,group in itertools.groupby(itertools.combinations(colors_lab, 2), key=lambda pair: pair[0]):
        group = list(group)
        #print lcol1, len(group), group[0]
        _,lcol2s = zip(*group)
        diffs = delta_e_cie2000(lcol1, np.array(lcol2s))
        col1 = colors[colors_lab.index(lcol1)]
        for lcol2,diff in zip(lcol2s, diffs):
            col2 = colors[colors_lab.index(lcol2)]
            pair = tuple(sorted([col1,col2]))
            pairdiffs[pair] = diff
    #print len(colors_lab)*len(colors_lab), pairdiffs.shape
    #PIL.Image.fromarray(pairdiffs).show()
    #fsdfs
    return pairdiffs

def color_differences_arr(colors):
    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
                  for col in colors]
    pairdiffs = np.zeros((len(colors),len(colors)))
    for lcol1,group in itertools.groupby(itertools.combinations(colors_lab, 2), key=lambda pair: pair[0]):
        group = list(group)
        #print lcol1, len(group), group[0]
        _,lcol2s = zip(*group)
        diffs = delta_e_cie2000(lcol1, np.array(lcol2s))
        col1 = colors[colors_lab.index(lcol1)]
        for lcol2,diff in zip(lcol2s, diffs):
            col2 = colors[colors_lab.index(lcol2)]
            pair = tuple(sorted([col1,col2]))
            pairdiffs[pair[0],pair[1]] = diff
    #print len(colors_lab)*len(colors_lab), pairdiffs.shape
    #PIL.Image.fromarray(pairdiffs).show()
    #fsdfs
    return pairdiffs

def color_edges(im, colorthresh=10):
    # see alternatively faster approx: https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter
    im_arr = np.array(im)
    
    # quantize and convert colors
    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
    quant_arr = np.array(quant)
    qcounts,qcolors = zip(*sorted(quant.getcolors(256), key=lambda x: x[0]))
    counts,colors = zip(*sorted(quant.convert('RGB').getcolors(256), key=lambda x: x[0]))

    # calc diffs
    pairdiffs = color_differences(colors)
    pairdiffs_arr = np.zeros((256,256))
    for k,v in list(pairdiffs.items()):
        qx,qy = (qcolors[colors.index(k[0])], qcolors[colors.index(k[1])])
        pairdiffs_arr[qx,qy] = v
    #PIL.Image.fromarray(pairdiffs_arr*50).show()

    # detect color edges
    orig_flat = np.array(quant).flatten()
    diff_im_flat = np.zeros(quant.size).flatten()
    for xoff in range(-1, 1+1, 1):
        for yoff in range(-1, 1+1, 1):
            if xoff == yoff == 0: continue
            off_flat = np.roll(quant, (xoff,yoff), (0,1)).flatten()
            diff_im_flat = diff_im_flat + pairdiffs_arr[orig_flat,off_flat] #np.maximum(diff_im_flat, pairdiffs[orig_flat,off_flat])
    diff_im_flat = diff_im_flat / 8.0

    diff_im_flat[diff_im_flat > colorthresh] = 255
    diff_im = diff_im_flat.reshape((im.height, im.width))

    #diff_im = diff_im_flat.reshape((im.height, im.width)).astype(np.uint8)
    #ret,diff_im = cv2.threshold(diff_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #print diff_im.min(), diff_im.mean(), diff_im.max()
    
    #quant.show()
    #PIL.Image.fromarray(diff_im).show()

    diff_im = diff_im / diff_im.max()
    #PIL.Image.fromarray(diff_im).show()
    diff_im *= 255
    #PIL.Image.fromarray(diff_im).show()
    diff_im = diff_im.astype(np.uint8)
    #PIL.Image.fromarray(diff_im).show()
    return diff_im

def color_changes(im):
##    im_arr = np.array(im)
##    im_arr = 256 * im_arr[:,:,0] + 256 * im_arr[:,:,1] + im_arr[:,:,2]
##
##    pairs = set()
##    for x in range(0+1, im_arr.shape[1]-1):
##        for y in range(0+1, im_arr.shape[0]-1):
##            col = im_arr[y,x]
##            
##            for xoff in range(-1, 1+1, 1):
##                for yoff in range(-1, 1+1, 1):
##                    if xoff == yoff == 0: continue
##                    col2 = im_arr[y+yoff,x+xoff]
##                    pair = (col,col2)
##                    pair = tuple(sorted(pair))
##                    pairs.add(pair)
##    print len(pairs)
##    return pairs

    im_arr = np.array(im)
    
    # quantize and convert colors
    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
    quant_arr = np.array(quant)
    qcounts,qcolors = zip(*sorted(quant.getcolors(256), key=lambda x: x[0]))
    counts,colors = zip(*sorted(quant.convert('RGB').getcolors(256), key=lambda x: x[0]))

    # calc diffs
    pairdiffs = color_differences(colors)
    pairdiffs_arr = np.zeros((256,256))
    for k,v in list(pairdiffs.items()):
        qx,qy = (qcolors[colors.index(k[0])], qcolors[colors.index(k[1])])
        pairdiffs_arr[qx,qy] = v
    #PIL.Image.fromarray(pairdiffs_arr*50).show()

    # detect color edges
    orig_flat = np.array(quant).flatten()
    diff_im_flat = np.zeros(quant.size).flatten()
    for xoff in range(-1, 1+1, 1):
        for yoff in range(-1, 1+1, 1):
            if xoff == yoff == 0: continue
            off_flat = np.roll(quant, (xoff,yoff), (0,1)).flatten()
            diff_im_flat = diff_im_flat + pairdiffs_arr[orig_flat,off_flat] #np.maximum(diff_im_flat, pairdiffs[orig_flat,off_flat])
    diff_im_flat = diff_im_flat / 8.0
    diff_im_flat[diff_im_flat < 10] = 0

    diff_im = diff_im_flat.reshape((im.height, im.width))
    diff_im = diff_im / diff_im.max()
    diff_im *= 255
    diff_im = diff_im.astype(np.uint8)
    
    diff_im = PIL.Image.fromarray(diff_im)
    return diff_im

def contrast(im):
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB)
    #cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return PIL.Image.fromarray(final)

def histogram(im):
    im = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
    im_arr = np.array(im)
    #im_arr = 256 * im_arr[:,:,0] + 256 * im_arr[:,:,1] + im_arr[:,:,2]

    colors,counts = np.unique(im_arr, return_counts=True)

    import matplotlib.pyplot as plt
    plt.hist(colors, colors, weights=counts)
    plt.show()

def group_colors(colors, thresh=10, categories=None):
    pairdiffs = color_differences(colors)
    diffgroups = dict([(c,[]) for c in categories]) if categories else dict()
    for c in colors:
        # find closest existing group that is sufficiently similar
        dists = [(gc,pairdiffs[tuple(sorted([c,gc]))])
                 for gc in diffgroups.keys()]
        similar = [(gc,dist) for gc,dist in dists if c != gc and dist < thresh]
        if similar:
            nearest = sorted(similar, key=lambda x: x[1])[0][0]
            diffgroups[nearest].append(c)
            # update that group key as the new central color (lowest avg dist to group members)
            gdists = [(gc1,[pairdiffs[tuple(sorted([c,gc]))] for gc2 in diffgroups[nearest]]) for gc1 in diffgroups[nearest]]
            central = sorted(gdists, key=lambda(gc,gds): sum(gds)/float(len(gds)))[0][0]
            diffgroups[central] = diffgroups.pop(nearest)
        else:
            diffgroups[c] = [c]
    return diffgroups

def view_colors(colors):
    import pyagg
    c=pyagg.Canvas(1000,200)
    c.percent_space()
    x = 2
    for i,g in enumerate(colors):
        print i, g
        x += 2
        c.draw_line([(x,0),(x,100)], fillcolor=g, fillsize=2)
    c.get_image().show()

def get_edge_colors(im):
    w,h = im.size
    arr = np.array(im)
    
    top = arr[0, :]
    bottom = arr[h-1, :]
    left = arr[:, 0]
    right = arr[:, w-1]

    edgecolors = np.vstack((top,bottom,left,right))
    edgecolors = np.unique(edgecolors, axis=0)
    return edgecolors

def rgb_to_lab(im):
    from PIL import Image, ImageCms
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    im = ImageCms.applyTransform(im, rgb2lab)
    return im

# --------------------------------------------------


if __name__ == '__main__':
    im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\cameroon.jpg")
    im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\repcongo.png")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\txu-pclmaps-oclc-22834566_k-2c.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\2113087.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\brazil_army_amazon_1999.jpg")
    im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\brazil_land_1977.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\israel-and-palestine-travel-reference-map-[2]-1234-p.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\algeria_rel79.jpg")
    im_orig = im

    # detect text colors
    def sniff_text_colors(im, samples=15):
        from PIL import ImageFilter
        from PIL import ImageOps
        from random import uniform
        w,h = im.size
        sw,sh = 200,200
        texts = []
        for i,(x,y) in enumerate(sample_quads(im, (sw,sh))):
            print '---'
            print 'sample',i,(x,y)
            sample = im.crop((x,y,x+sw,y+sh))
            lab = rgb_to_lab(sample)
            l,a,b = lab.split()
            #print np.array(l).min(), np.array(l).max()
            lup = l.resize((l.size[0]*2,l.size[1]*2), PIL.Image.LANCZOS)
            #lup.show()
            data = automap.main.detect_data(lup)
            for text in data:
                #print '---',text
                if float(text['conf']) > 50 and len(text['text']) >= 3:
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
                    print 'FOUND',text
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
                    ls = 1 - ((ls.flatten()-ls.min()) / float(ls.max()))
                    ls[ls < 0.5] = 0
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
                    
                    print textcol
                    texts.append((text['text'],textcol))
            if i >= 3 and len(texts) > samples:
                break
        textcolors = [t[1] for t in texts]
        textcolors = group_colors(textcolors, 20)
        print 'textcolors detected',textcolors
        return textcolors

    def sample_quads(im, tilesize):
        import random
        w,h = im.size
        tw,th = tilesize

        # divide image into quad tiles
        quads = []
        for x in range(0, w-tw, tw):
            for y in range(0, h-th, th):
                quads.append((x,y))

        # randomly choose a quad from each corner until completion
        while quads:
            # nw
            corner = [q for q in quads if q[0] < w/2.0 and q[1] < h/2.0]
            if corner:
                q = random.choice(corner)
                quads.remove(q)
                yield q
            # ne
            corner = [q for q in quads if q[0] > w/2.0 and q[1] < h/2.0]
            if corner:
                q = random.choice(corner)
                quads.remove(q)
                yield q
            # se
            corner = [q for q in quads if q[0] > w/2.0 and q[1] > h/2.0]
            if corner:
                q = random.choice(corner)
                quads.remove(q)
                yield q
            # sw
            corner = [q for q in quads if q[0] < w/2.0 and q[1] > h/2.0]
            if corner:
                q = random.choice(corner)
                quads.remove(q)
                yield q

    def sample_texts(im, textcolors, samplesize=(300,300)):
        from random import uniform
        import re
        w,h = im.size
        sw,sh = samplesize
        texts = []
        # for each sample
##        for _ in range(samples):
##            print _
##            x,y = uniform(0,w-sw),uniform(0,h-sh)
        for i,(x,y) in enumerate(sample_quads(im, (sw,sh))):
            print '---'
            print 'sample',i,(x,y)
            sample = im.crop((x,y,x+sw,y+sh))
            # upscale
            print 'upscaling'
            upscale = sample.resize((sample.size[0]*2,sample.size[1]*2), PIL.Image.LANCZOS)
            upscale = quantize(upscale)
            #upscale.show()
            for col in textcolors:
                # isolate color
                print 'isolating color'
                diff = color_difference(upscale, col)
                #diff.show()
                # detect text
                print 'running ocr'
                data = automap.main.detect_data(diff)
                print 'processing text'
                for text in data:
                    # process text
                    if float(text['conf']) > 50 and len(text['text']) >= 2:
                        # clean text
                        text['text'] = re.sub('^\\W+|\\W+$', '', text['text'], flags=re.UNICODE) # strips nonalpha chars from start/end
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
                        # record info
                        text['color'] = col
                        # downscale coords
                        for key in 'left top width height'.split():
                            text[key] = float(text[key]) / 2.0
                        # ignore edge cases
                        edgebuff = text['height']
                        if text['left'] < edgebuff or text['top'] < edgebuff \
                           or (text['left']+text['width']) > sw-edgebuff or (text['top']+text['height']) > sh-edgebuff:
                            #print 'edge case',text
                            #print [edgebuff,edgebuff,sw-edgebuff,sh-edgebuff]
                            continue
                        # convert sample space to image space
                        text['left'] = x + float(text['left'])
                        text['top'] = y + float(text['top'])
                        texts.append(text)

            print 'texts',len(texts)
            if i >= 3 and len(texts) > 30:
                break
                    
        return texts

    textcolors = sniff_text_colors(im)
    view_colors(textcolors)

    # compare with just luminance
##    lab = rgb_to_lab(im)
##    l,a,b = lab.split()
##    l.show()
##    im = l
##    textcolors = [(0,0,0)]

    from time import time
    t=time()

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
    texts = sample_texts(im, textcolors)
    print time()-t
    
##    for t in texts:
##        print t
        
    import pyagg
    c = pyagg.canvas.from_image(im)
    c.pixel_space()
    for t in texts:
        left,top,width,height = [t[k] for k in 'left top width height'.split()]
        c.draw_box(bbox=[left,top,left+width,top+height], fillcolor=None, outlinecolor=(0,255,0), outlinewidth='2px')
        c.draw_text(t['text'], xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0))
    c.get_image().show()

##    # find colors around edges
##    from PIL import ImageFilter
##    background = im.filter(ImageFilter.MedianFilter(5))
##    background.show()
##
##    edgecolors = get_edge_colors(quantize(background))
##    # TODO: also consider counts of each color to get most common, > 95% after color grouping
##    edgecolors = [tuple(col) for col in edgecolors]
##    print 'grouping'
##    edgecolors = group_colors(edgecolors, 20)
##    view_colors(edgecolors)
##
##    if len(edgecolors) > 1:
##        raise Exception
##
##    edgecolor = list(edgecolors.keys())[0]
##
##    diff = color_difference(quantize(im), edgecolor)
##    diff = diff.point(lambda v: 255 if v > 10 else 0)
##    diff.show()
##
##    edges = color_edges(quantize(im), 5)
##    PIL.Image.fromarray(edges).show()
##
##    # TODO: Maybe grow the diff region before tracing the contours
##
##    fdasfs












            

    
