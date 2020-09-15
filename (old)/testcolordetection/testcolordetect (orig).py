
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
    for y in range(im.size[0]):
        for x in range(im.size[1]):
            c = imar[y,x]
            c2 = imar2[y,x]
            d = delta_e_cie2000(convert_color(sRGBColor(*c, is_upscaled=True), LabColor).get_value_tuple(),
                                np.array([convert_color(sRGBColor(*c2, is_upscaled=True), LabColor).get_value_tuple()]),
                                )
            diffs[y,x] = d
    return diffs

def color_difference(im, color):
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

    #dissim = diff_im > thresh
    #im_arr[dissim] = (255,255,255)
    #diff_im[dissim] = 255
    im_arr = diff_im.astype(np.uint8)
    #PIL.Image.fromarray(im_arr).show()
    #fdsf

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

# --------------------------------------------------


if __name__ == '__main__':
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\cameroon.jpg")
    im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\repcongo.png")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\brazil_land_1977.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\txu-pclmaps-oclc-22834566_k-2c.jpg").crop((1000,1000,2000,2000))
    im_orig = im
    im = smooth(im)
    im.show()

    # edges
##    im = im.resize((im.size[0]*4,im.size[1]*4), PIL.Image.NEAREST)
####    im.show()
####    diff = PIL.Image.fromarray(color_edges(im, 10))
####    diff.show()
##    diff = color_changes(im)
##    diff.show()
##    _,thresh = cv2.threshold(np.array(diff),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    thresh = PIL.Image.fromarray(thresh)
##    thresh.show()
####    kernel = np.ones((3,3), np.uint8)
####    thresh = cv2.dilate(np.array(thresh), kernel, iterations=1)
####    thresh = PIL.Image.fromarray(thresh)
####    thresh.show()
##    contours,_ = cv2.findContours(np.array(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
##    im_arr = np.array(im)
##    for c in contours:
##        mask = np.zeros(im_arr.shape, np.uint8)
##        cv2.drawContours(mask, c, -1, 255, -1)
##        mean = cv2.mean(im_arr, mask=mask)
##        print mean

    # maincolors
##    im.show()
##    #im = im.resize((im.size[0]/5,im.size[1]/5), PIL.Image.ANTIALIAS)
##    print im
##    im.show()
##    maincolors(im)

    # color histogram
##    histogram(im)

    # contrast
##    im = contrast(im)
##    im.show()

    # kmeans
    #from PIL import ImageFilter
    #im = im.filter(ImageFilter.MinFilter(5))
##    colors = np.array(im).reshape((-1,3)).astype(np.float32)
##    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##    K = 8
##    ret,labels,centers = cv2.kmeans(colors,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
##    centers = [map(int, c) for c in centers]
##    view_colors(centers)
##
##    im_prep = im_orig #smooth(im_orig) # not sure...
##    im_prep = im_prep.resize((im_prep.size[0]*2, im_prep.size[1]*2), PIL.Image.LANCZOS)
##    im_prep = quantize(im_prep)
##    for c in sorted(centers, key=lambda rgb: sum(rgb)/3.0):
##        print c
##        t,mask = threshold(im_prep, c, 20) 
##        t.show()
##        break

    # kmeans with regional
    colors = np.array(im).reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,labels,centers = cv2.kmeans(colors,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centers = [map(int, c) for c in centers]
    sortcenters = sorted(centers, key=lambda rgb: sum(rgb)/3.0)
    view_colors(sortcenters)

    # reconstruct according to kmeans
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res2 = res.reshape((im.size[1],im.size[0],3))
    PIL.Image.fromarray(res2).show()

    # or group based on dist
##    im_prep = quantize(im)
##    diff = np.ones((im.size[1],im.size[0]))*255
##    for c in centers:
##        print c
##        d = color_difference(im_prep, c) 
##        d.show()
##        #break
##    fdsf

    # TODO:
    # http://www.cse.msu.edu/~pramanik/research/papers/2002Papers/icip.hsv.pdf
    # https://stackoverflow.com/questions/30483886/kmeans-clustering-on-different-distance-function-in-lab-space
    # maybe convert to hsv or lab space before kmeans...
    # maybe eliminate unnecessary categories by merging those that are too similar? 
    # of the identified colors, assign each pixel to the most similar category
    
    im_prep = im #smooth(im_orig) # not sure...
    im_prep = im_prep.resize((im_prep.size[0]*2, im_prep.size[1]*2), PIL.Image.LANCZOS)
    im_prep = quantize(im_prep)
    data = []
    for c in sortcenters:
        print c
        t,mask = threshold(im_prep, c, 20)
        t.show()

        # ocr
        print 'detecting text'
        subdata = detect_data(t)
        print 'processing text'
        subdata = process_text(subdata, 50)

        for r in subdata:
            print r['text']

        # assign text characteristics
        for dct in subdata:
            dct['color'] = c

        # filter out duplicates from previous loop, in case includes some of the same pixels
        print '(skip duplicates)',len(subdata),len(data)
        subdata = [r for r in subdata
                   if (r['top'],r['left'],r['width'],r['height'])
                   not in [(dr['top'],dr['left'],dr['width'],dr['height']) for dr in data]
                   ]

        # connect text data
        print '(connecting texts)'
        subdata = connect_text(subdata)

        data.extend(subdata)

##    from PIL import ImageFilter
##    background = im.filter(ImageFilter.MaxFilter(5))
##    diff = color_difference_ims(im, background)
##    diff.show()

    # detect anchors

    # assign function

    # done, list
    for r in data:
        print r['text']
    
    fsdf















    # foreground/background
    from PIL import ImageFilter
    #im_arr = np.array(im)
    #im_arr = 256 * im_arr[:,:,0] + 256 * im_arr[:,:,1] + im_arr[:,:,2]
    #im = PIL.Image.fromarray(im_arr)
    #im = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
    #im = im.filter(ImageFilter.ModeFilter(3*10))
    im = im.filter(ImageFilter.MinFilter(5))
    #im = smooth(im)
    im.show()
    
    thresh = PIL.Image.fromarray(color_edges(im, 5))
    thresh.show()
    
##    diff = color_changes(im)
##    diff.show()
##    _,thresh = cv2.threshold(np.array(diff),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    thresh = PIL.Image.fromarray(thresh)
##    thresh.show()

    from PIL import ImageStat
    contours,hierarchy = cv2.findContours(np.array(thresh), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
    im_arr = np.array(im)
    print len(contours)
    means = set()
    for c in contours:
        mask = np.zeros(im_arr.shape[:2], np.uint8)
        cv2.drawContours(mask, c, -1, 255, -1)
        mask = PIL.Image.fromarray(mask)
        #mean = cv2.mean(im_arr, mask=mask)
        stats = ImageStat.Stat(im, mask)
        stdev = stats.stddev
        meanstdev = sum(stdev)/3.0
        if meanstdev < 5:
            mean = stats.mean
            mean = tuple([int(round(round(v/255.0, 2)*255)) for v in mean])
            #print mean
            means.add(mean)
            
    print len(means)

    # group contour colors
    diffgroups = group_colors(list(means))
    view_colors(diffgroups.keys())

    # group color groupings
    diffgroups2 = group_colors(diffgroups.keys(), thresh=20)
    view_colors(diffgroups2.keys())

    im_prep = smooth(im_orig) # not sure...
    im_prep = im_prep.resize((im_prep.size[0]*2, im_prep.size[1]*2), PIL.Image.LANCZOS)
    im_prep = quantize(im_prep)
    for c in sorted(diffgroups2.keys(), key=lambda rgb: sum(rgb)/3.0):
        print c
        t,mask = threshold(im_prep, c, 30) # or on im, for singling out the minfilter colors
        t.show()
        break

    # maybe calc diff from local minfilter instead of same global color? 

    # maybe approx rectangles from local minfilter, and use horizontal boxes w repeating heights as text/toponym color

    








            

    
