
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
    #diff_im_flat[diff_im_flat < 10] = 0

    diff_im = diff_im_flat.reshape((im.height, im.width))
    #diff_im = diff_im / diff_im.max()
    #diff_im *= 255
    #diff_im = diff_im.astype(np.uint8)
    
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

def local_homogeneity(im):
    im_arr = np.array(im)
    
    # quantize and convert colors
    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
    quant_arr = np.array(quant)
    qcounts,qcolors = zip(*sorted(quant.getcolors(256), key=lambda x: x[0]))
    counts,colors = zip(*sorted(quant.convert('RGB').getcolors(256), key=lambda x: x[0]))

    # calc homogeneity bw two colors as zfuncs of color distance
    def zfunc(dist):
        dmin = 0
        dmax = 150
        dist = min(dist,dmax) # cap it
        dwidth = float(dmax-dmin)
        if dist < dmax/2.0:
            return 1 - 2*((dist-dmin)/dwidth)**2
        else:
            return 2*((dist-dmax)/dwidth)**2
    pairdiffs = color_differences(colors)
    pairdiffs_arr = np.zeros((256,256))
    for k,v in list(pairdiffs.items()):
        qx,qy = (qcolors[colors.index(k[0])], qcolors[colors.index(k[1])])
        pairdiffs_arr[qx,qy] = zfunc(v)

    # loop neighbours and assign precalc homogeneity, and take avg of all neighbours
    orig_flat = np.array(quant).flatten()
    diff_im_flat = np.zeros(quant.size).flatten()
    for xoff in range(-1, 1+1, 1):
        for yoff in range(-1, 1+1, 1):
            if xoff == yoff == 0: continue
            off_flat = np.roll(quant, (xoff,yoff), (0,1)).flatten()
            diff_im_flat = diff_im_flat + pairdiffs_arr[orig_flat,off_flat] #np.maximum(diff_im_flat, pairdiffs[orig_flat,off_flat])
    diff_im_flat = diff_im_flat / 8.0

    diff_im = diff_im_flat.reshape((im.height, im.width))

    return diff_im

def find_prototype_colors(im):
    nullcolor = (255,255,255)

    # maybe prep as just foreground pixels???
##    from PIL import ImageFilter
##    im = im.filter(ImageFilter.MinFilter(3))

    # PURE COLOR DISTS
##    # local color distance
##    homogthresh = 20
##    changes = color_changes(im)
##    changes.show()
##
##    # pick only homogenous pixels with low distances
##    changearr = np.array(changes)
##    homogmask = changearr > homogthresh # opposite bc want to reset where not homogenous
##    homogim = np.array(im)
##    homogim[homogmask] = nullcolor
##    homogim = PIL.Image.fromarray(homogim)
##    homogim = quantize(homogim)
##    homogim.show()

    # OR COLOR HOMOGENEITY ZFUNC
    # local color homogeinity zfunc vals (0-1)
    homogthresh = 0.75
    zvals = local_homogeneity(im)
    PIL.Image.fromarray(zvals*255).convert('L').show()

    # pick only homogenous pixels with low values
    zvals = np.array(zvals)
    homogmask = zvals < homogthresh # opposite bc want to reset where not homogenous
    homogim = np.array(im)
    homogim[homogmask] = nullcolor
    homogim = PIL.Image.fromarray(homogim)
    homogim = quantize(homogim)
    homogim.show()

    # OR just do simple foreground pixels and then select prototypes from that???
##    from PIL import ImageFilter
##    homogim = im.filter(ImageFilter.MinFilter(3))
##    homogim = quantize(homogim)
##    homogim.show()

    countthresh = 100
    similthresh = 10
    protos = []
    while True:
        # extract prototype
        histo = homogim.getcolors(homogim.size[0]*homogim.size[1])
        histo = [(cn,px) for cn,px in histo if px != nullcolor]
        histo = sorted(histo, key=lambda(cn,px): -cn)
        if not histo:
            break
        count,proto = histo[0]
        if count < countthresh:
            break
        print proto,count
        protos.append(proto)

        # remove similar prototypes
        diff = color_difference(homogim, proto)
        diff = np.array(diff)
        homogim = np.array(homogim)
        homogim[diff < similthresh] = nullcolor
        homogim = PIL.Image.fromarray(homogim)

    return protos

def assign_to_prototypes(im, prototypes):
    im = quantize(im)
    nearestproto = np.array(im).reshape((im.size[1]*im.size[0], 3))
    nearestproto[:,:] = (255,255,255)
    
    diffmin = None
    for proto in prototypes:
        print proto

        diff = color_difference(im, proto)
        diff = np.array(diff).flatten()

        if diffmin is not None:
            wheremin = (diff < diffmin)
            nearestproto[wheremin] = proto
            diffmin = np.minimum(diff, diffmin)
            
        else:
            diffmin = diff
            nearestproto[:,:] = proto

    nearestproto = nearestproto.reshape((im.size[1], im.size[0], 3))
    nearestproto = PIL.Image.fromarray(nearestproto)
    return nearestproto

# --------------------------------------------------


if __name__ == '__main__':
    im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\cameroon.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\repcongo.png")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\brazil_land_1977.jpg")
    #im = PIL.Image.open(r"C:\Users\kimok\OneDrive\Documents\GitHub\AutoMap\tests\testmaps\txu-pclmaps-oclc-22834566_k-2c.jpg")
    im_orig = im
    im_orig.show()

    #im = smooth(im)
    
##    from PIL import ImageFilter
##    im = im.filter(ImageFilter.MinFilter(5))
##    im.show()

    # dbscan
    from sklearn.cluster import MeanShift
    #colors = np.array(im).reshape((-1,3)).astype(np.float32)
    colorcounts = im.getcolors(im.size[0]*im.size[1])
    #maxcount = max(colorcounts, key=lambda(cn,col):cn)
    counts = np.array([cn for cn,col in colorcounts])
    colors = np.array([col for cn,col in colorcounts])
    print colors
    print counts

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _3dcolors = np.array([col for cn,col in colorcounts if cn > 5])
    _3dcounts = np.array([cn for cn,col in colorcounts if cn > 5])
    print len(_3dcolors)
    rs,gs,bs = zip(*_3dcolors)
    #ax.scatter(rs, gs, bs, c=_3dcolors/255.0)#, s=_3dcounts/1000.0)
    #plt.show()

    res = MeanShift(bandwidth=33, bin_seeding=True).fit(_3dcolors) #, sample_weight=_3dcounts)
    print res
    protos = []
    for lab in set(res.labels_):
        if lab == -1: continue
        #labcolors = [_3dcolors[i] for i,l in enumerate(res.labels_) if l==lab]
        labcolors = _3dcolors[res.labels_==lab]
        print lab,len(labcolors)
        #view_colors([tuple(rgb) for rgb in labcolors[:20]])
        rs,gs,bs = zip(*labcolors)
        r = sum(rs)/float(len(rs))
        g = sum(gs)/float(len(gs))
        b = sum(bs)/float(len(bs))
        r,g,b = map(int, (r,g,b))
        protos.append((r,g,b))
    
    # view colors
    print len(protos)
    view_colors(protos)

    # assign pixels to main colors
    assigned = assign_to_prototypes(im, protos)
    assigned.show()
    
    fdsfsd











            

    
