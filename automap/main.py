
from . import geocode
from .triangulate import triangulate, triangulate_add
from .shapematch import normalize
from .rmse import optimal_rmse, polynomial

import os
import itertools
import tempfile
import time
import re

import pythongis as pg

import pytesseract as t

import PIL, PIL.Image

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff_matrix import delta_e_cie2000

import numpy as np
import cv2


def threshold(im, color, thresh):
    im_arr = np.array(im)

    #target = color
    #diffs = [pairdiffs[tuple(sorted([target,oth]))] for oth in colors if oth != target]

    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
    counts,colors = zip(*im.getcolors(256))
    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
                  for col in colors]
    diffs = delta_e_cie2000(target, np.array(colors_lab))
    
    difftable = dict(list(zip(colors,diffs)))
    diff_im = np.zeros((im.height,im.width))
    for col,diff in difftable.items():
        #print col
        diff_im[(im_arr[:,:,0]==col[0])&(im_arr[:,:,1]==col[1])&(im_arr[:,:,2]==col[2])] = diff # 3 lookup is slow

    dissim = diff_im > thresh
    #im_arr[dissim] = (255,255,255)
    diff_im[dissim] = 255
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
    return im, dissim

def quantize(im):
    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256).convert('RGB')
    return quant

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

def maincolors(im):
    import pyagg

    # View all colors
##    c = pyagg.graph.BarChart()
##    bins = im.getcolors(im.size[0]*im.size[1])
##    for cn,col in sorted(bins, key=lambda b: -b[0])[:10]:
##        #print cn,col
##        c.add_category(col, [(col,cn)]) #, fillcolor=col)
##    c.draw().view()
    
##    c = pyagg.Canvas(1200, 500)
##    c.custom_space(0, 500, 1200, 0)
##    bins = im.getcolors(im.size[0]*im.size[1])
##    print len(bins)
##    bins = [b for b in bins if b[0] > 1]
##    print len(bins)
##    maxval = max((b[0] for b in bins))
##    x = 0
##    bins = sorted(bins, key=lambda b: -b[0])[:1000]
##    incr = c.width/float(len(bins))
##    for cn,col in bins:
##        c.draw_box(bbox=[x,0,x+incr,cn/float(maxval)*c.height], fillcolor=col)
##        x += incr
##    c.view()

    # Using lab dist... (old)
##    import numpy as np
##    from skimage.color import rgb2lab
##    
##    lab_im_arr = rgb2lab(np.array(im))
##    print lab_im_arr
##    colors,counts = np.unique(lab_im_arr.reshape(im.size+(3,)), return_counts=True)
##    for cn,col in zip(colors,counts):
##        print cn,col

    # Altern, use kmeans clustering...
    # https://www.alanzucconi.com/2015/05/24/how-to-find-the-main-colours-in-an-image/
    # see also https://www.alanzucconi.com/2015/09/30/colour-sorting/
##    from sklearn.cluster import KMeans
##    from sklearn.metrics import silhouette_score
##
##    # By Adrian Rosebrock
##    import numpy as np
##    import cv2
##
##    def centroid_histogram(clt):
##        # grab the number of different clusters and create a histogram
##        # based on the number of pixels assigned to each cluster
##        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
##        (hist, _) = np.histogram(clt.labels_, bins = numLabels)
##     
##        # normalize the histogram, such that it sums to one
##        hist = hist.astype("float")
##        hist /= hist.sum()
##     
##        # return the histogram
##        return hist
##
##    # Reshape the image to be a list of pixels
##    im = im.resize((im.size[0]/50, im.size[1]/50))
##    image_array = np.array(im).reshape((im.size[0] * im.size[1], 3))
##    print len(image_array)
##
##    def drawhist(bins):
##        c = pyagg.Canvas(1200, 500)
##        c.custom_space(0, 500, 1200, 0)
##        maxval = max((b[0] for b in bins))
##        x = 0
##        bins = sorted(bins, key=lambda b: -b[0])[:1000]
##        incr = c.width/float(len(bins))
##        for cn,col in bins:
##            c.draw_box(bbox=[x,0,x+incr,cn/float(maxval)*c.height], fillcolor=tuple(col))
##            x += incr
##        c.view()
##
##    # Clusters the pixels
##    clt = KMeans(n_clusters=classes)
##    clt.fit(image_array)
##
##    # Finds how many pixels are in each cluster
##    hist = centroid_histogram(clt)
##
##    bins = list(zip(hist, clt.cluster_centers_))
##    for perc,col in bins:
##        print perc,col
##
##    drawhist(bins)
##
####    bestSilhouette = -1
####    bestClusters = 0
####
####    for clusters in range(3, 5): 
####        print 'clusters',clusters
####        
####        # Cluster colours
####        clt = KMeans(n_clusters = clusters)
####        clt.fit(image_array)
####
####        # Validate clustering result
####        silhouette = silhouette_score(image_array, clt.labels_, metric='euclidean')
####
####        # Find the best one
####        if silhouette > bestSilhouette:
####            bestSilhouette = silhouette
####            bestClusters = clusters

    # EXPERIMENTAL GLOBAL COLOR DETECTION
##    from colormath.color_objects import sRGBColor, LabColor
##    from colormath.color_conversions import convert_color
##    from colormath.color_diff_matrix import delta_e_cie2000
##
##    import numpy as np
##
##    im_arr = np.array(im)
##
##    # quantize and convert colors
##    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
##    qcolors = [col for cn,col in sorted(quant.getcolors(256), key=lambda e: e[0])]
##    colors = [col for cn,col in sorted(quant.convert('RGB').getcolors(256), key=lambda e: e[0])]
##    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
##                  for col in colors]
##
##    # calc diffs
##    pairdiffs = dict()
##    for col,lcol in zip(colors,colors_lab):
##        diffs = delta_e_cie2000(lcol, np.array(colors_lab))
##        for col2,diff in zip(colors,diffs):
##            if col != col2 and diff <= 10:
##                pairdiffs[(col,col2)] = diff
##    print len(colors_lab)*len(colors_lab), len(pairdiffs)
####    import itertools
####    pairdiffs = dict()
####    for (col,lcol),(col2,lcol2) in itertools.combinations(zip(colors,colors_lab), 2):
####        diff = delta_e_cie2000(lcol, np.array([lcol2]))
####        if diff <= 10:
####            pairdiffs[(col,col2)] = diff
####    print len(colors_lab)*len(colors_lab), len(pairdiffs)
##    
##    # group custom
##    diffgroups = []
##    popcolors = list(colors)
##    while popcolors:
##        pop = popcolors.pop(0)
##        diffgroup = [pop]
##        for pair in pairdiffs.keys():
##            if pair[0]==pop:
##                conn = pair[1]
##                diffgroup.append(conn)
##                try: popcolors.pop(popcolors.index(conn))
##                except: pass
##        if len(diffgroup) > 1:
##            diffgroups.append(diffgroup)
##    diffgroups = set([tuple(sorted(g)) for g in diffgroups])
##    for i,g in enumerate(diffgroups):
##        print i, g[0], len(g)
##
##    # group network graph
####    import networkx as nx
####    graph = nx.Graph()
####    graph.add_edges_from(pairdiffs)
####    groups = nx.connected_components(graph)
####    for i,g in enumerate(groups):
####        print i, list(g)[0], len(g)
##
##    # view
##    import pyagg
##    c=pyagg.Canvas(1000,200)
##    c.percent_space()
##    x = 0
##    for i,g in enumerate(diffgroups):
##        x += 1
##        for col in g:
##            x += 0.1
##            c.draw_line([(x,0),(x,100)], fillcolor=col, fillsize=0.1)
##    c.view()
##    fdsfds

    # EXPERIMENTAL COLOR AREAS
    colorthresh = 5
    
    im_arr = np.array(im)

    # compare w normal binarization
##    import cv2
##    im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
##    im_arr = cv2.GaussianBlur(im_arr,(3,3),0)
##    ret,im_arr = cv2.threshold(im_arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    PIL.Image.fromarray(im_arr).show()

    # smooth small color changes
    import cv2
    #im_arr = cv2.GaussianBlur(im_arr,(3,3),0)
    #im = PIL.Image.fromarray(im_arr)

    # quantize and convert colors
    quant = quantize(im)
    counts,colors = zip(*quant.getcolors(256))

    # calc diffs
    pairdiffs = color_differences(colors)

    # detect color edges
##    orig_flat = np.array(quant).flatten()
##    diff_im_flat = np.zeros(quant.size).flatten()
##    for xoff in range(-1, 1+1, 1):
##        for yoff in range(-1, 1+1, 1):
##            if xoff == yoff == 0: continue
##            off_flat = np.roll(quant, (xoff,yoff), (0,1)).flatten()
##            diff_im_flat = diff_im_flat + pairdiffs[orig_flat,off_flat] #np.maximum(diff_im_flat, pairdiffs[orig_flat,off_flat])
##    diff_im_flat = diff_im_flat / 8.0
##
##    diff_im_flat[diff_im_flat > colorthresh] = 255
##    diff_im = diff_im_flat.reshape((im.height, im.width))
##
##    #diff_im = diff_im_flat.reshape((im.height, im.width)).astype(np.uint8)
##    #ret,diff_im = cv2.threshold(diff_im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    
##    print diff_im.min(), diff_im.mean(), diff_im.max()
##    quant.show()
##    PIL.Image.fromarray(diff_im).show()

    # group custom
##    pairdiffs = dict([((colors[qcolors.index(c1)],colors[qcolors.index(c2)]),pairdiffs[c1,c2])
##                      for c1 in range(pairdiffs.shape[0])
##                      for c2 in range(pairdiffs.shape[1])
##                      if pairdiffs[c1,c2] < 2])
##    print len(pairdiffs)
####    diffgroups = []
####    popcolors = list(colors)
####    while popcolors:
####        pop = popcolors.pop(0)
####        diffgroup = [pop]
####        for pair in pairdiffs.keys():
####            if pair[0]==pop:
####                conn = pair[1]
####                diffgroup.append(conn)
####                try: popcolors.pop(colors.index(conn))
####                except: pass
####        if len(diffgroup) > 1:
####            diffgroups.append(diffgroup)
####    diffgroups = set([tuple(sorted(g)) for g in diffgroups])
####    for i,g in enumerate(diffgroups):
####        print i, g[0], len(g)
##
##    # group network graph
##    import networkx as nx
##    graph = nx.Graph()
##    graph.add_edges_from(pairdiffs)
##    diffgroups = list(nx.connected_components(graph))
##    for i,g in enumerate(diffgroups):
##        print i, list(g)[0], len(g)

    # TODO: include count weights when calculating and assigning to main groups
    # ...

    # group custom 2
    colorthresh = 10
    
    diffgroups = dict()
    for c in colors:
        # find closest existing group that is sufficiently similar
        dists = [(gc,pairdiffs[tuple(sorted([c,gc]))]) for gc in diffgroups.keys()]
        similar = [(gc,dist) for gc,dist in dists if c != gc and dist < colorthresh]
        if similar:
            nearest = sorted(similar, key=lambda x: x[1])[0][0]
            diffgroups[nearest].append(c)
            # update that group key as the new central color (lowest avg dist to group members)
            gdists = [(gc1,[pairdiffs[tuple(sorted([c,gc]))] for gc2 in diffgroups[nearest]]) for gc1 in diffgroups[nearest]]
            central = sorted(gdists, key=lambda(gc,gds): sum(gds)/float(len(gds)))[0][0]
            diffgroups[central] = diffgroups.pop(nearest)
        else:
            diffgroups[c] = [c]
    # maybe also group color groupings together...
##    diffgroups2 = dict()
##    for c in diffgroups.keys():
##        # find closest existing group that is sufficiently similar
##        dists = [(gc,pairdiffs[c,gc]) for gc in diffgroups2.keys()]
##        similar = [(gc,dist) for gc,dist in dists if c != gc and dist < colorthresh]
##        if similar:
##            nearest = sorted(similar, key=lambda x: x[1])[0][0]
##            diffgroups2[nearest].append(c)
##            # update that group key as the new central color (lowest avg dist to group members)
##            gdists = [(gc1,[pairdiffs[gc1,gc2] for gc2 in diffgroups2[nearest]]) for gc1 in diffgroups2[nearest]]
##            central = sorted(gdists, key=lambda(gc,gds): sum(gds)/float(len(gds)))[0][0]
##            diffgroups2[central] = diffgroups2.pop(nearest)
##        else:
##            diffgroups2[c] = [c]
##    diffgroups = dict([(k,sum([diffgroups[gc] for gc in g],[])) for k,g in diffgroups2.items()])

    # for alternative groupings, see https://scikit-learn.org/stable/modules/clustering.html
    # ...

    # maybe cluster neighbouring color pixels based on most similar color/class in regional neighbourhood
    # ...

    # view
    if 1:
        import pyagg
        c=pyagg.Canvas(1000,200)
        c.percent_space()
        x = 2
        for i,g in enumerate(diffgroups.keys()):
            print i, g
            x += 2
            c.draw_line([(x,0),(x,100)], fillcolor=g, fillsize=2)
        c.get_image().show()
        
        c=pyagg.Canvas(1000,200)
        c.percent_space()
        x = 0
        for i,g in enumerate(diffgroups.values()):
            print i, list(g)[0], len(g)
            x += 1
            for col in g:
                x += 0.3
                c.draw_line([(x,0),(x,100)], fillcolor=col, fillsize=0.3)
        #c.get_image().show()

    if 1:
        # view colors in image
        #im.show()
        quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
        quant.show()
        qcounts,qcolors = zip(*sorted(quant.getcolors(256), key=lambda x: x[0]))
        counts,colors = zip(*sorted(quant.convert('RGB').getcolors(256), key=lambda x: x[0]))
        qarr = np.array(quant)
##        for k,g in diffgroups_dict.items():
##            colim = im_arr.copy()
##            
##    ##        colim[np.isin(qarr, g, invert=True)] = [255,255,255] #[0,0,0]
##            
##            diffs = [pairdiffs[k,oth] for oth in qcolors]
##            difftable = dict(list(zip(qcolors,diffs)))
##            diff_im_flat = np.array(quant).flatten()
##            for qcol,diff in difftable.items():
##                diff_im_flat[diff_im_flat==qcol] = diff
##            diff_im = diff_im_flat.reshape((im.height, im.width))
##            dissim = diff_im > 10
##            colim[dissim] = (255,255,255)
##
##            PIL.Image.fromarray(colim).show()
        colim = np.zeros((im_arr.shape[0]*im_arr.shape[1]*3,), np.uint8).reshape(im_arr.shape)
        for k,g in diffgroups.items():
            qg = [qcolors[colors.index(gs)] for gs in g]
            colim[np.isin(qarr, qg)] = k
        PIL.Image.fromarray(colim).show()
    
    return diffgroups

def maincolors(im):
    im_arr = np.array(quantize(im))

    # try downsizing to get avg colors?
    quantize(im).resize((500,500), PIL.Image.ANTIALIAS).show()

    # try upsampling 4 times and get stdev
    upsc = im.resize((im.size[0]*4,im.size[1]*4), PIL.Image.NEAREST)
    upsc.show()
    PIL.Image.fromarray(color_edges(upsc)).show()

    # detect color edges
    thresh = 10
    uniq = set()
    for xoff in range(-1, 0+1, 1):
        for yoff in range(-1, 0+1, 1):
            print xoff,yoff
            if xoff == yoff == 0: continue
            off = np.roll(im_arr, (xoff,yoff), (0,1))
            stack = np.stack([im_arr, off], axis=1)
            print im_arr.shape, off.shape, stack.shape
            #print stack
            #pairs,counts = np.unique(stack.reshape(im_arr.shape[0]*im_arr.shape[1], 6), return_counts=True, axis=1)
            #print pairs.shape, pairs
            #print counts.shape, counts
            uniq.update(set((tuple(v) for v in stack.reshape(im_arr.shape[0]*im_arr.shape[1], 6))))
            print len(uniq)

##    pairdiffs = dict()
##    for c,colors in itertools.groupby(sorted(uniq, key=lambda x: x[:3]), key=lambda x: x[:3]):
##        colors = [col[3:6] for col in colors]
##        c_lab = convert_color(sRGBColor(*c, is_upscaled=True), LabColor).get_value_tuple()
##        colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
##                      for col in colors]
##        diffs = delta_e_cie2000(c_lab, np.array(colors_lab))
##        simil = [(col,diff) for col,diff in zip(colors,diffs) if diff < thresh]
##        #print c, len(colors), len(simil)
##        pairdiffs.update([(tuple(sorted((c,col))),diff) for col,diff in simil])
##        #for col,diff in simil:
##        #    pairdiffs[tuple(sorted((c,col)))] = diff
    colors = list(set([col[3:6] for col in uniq]))
    pairdiffs = color_differences(colors)
    print len(pairdiffs),pairdiffs.items()[0]

    diffgroups = dict()
    for c in colors:
        # find closest existing group that is sufficiently similar
        #print c, diffgroups.keys()
        #c_lab = convert_color(sRGBColor(*c, is_upscaled=True), LabColor).get_value_tuple()
        #dists = [(gc,pairdiffs.get(tuple(sorted([c,gc]))) or delta_e_cie2000(c_lab, np.array([convert_color(sRGBColor(*gc, is_upscaled=True), LabColor).get_value_tuple()])))
        #         for gc in diffgroups.keys()]
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

    # view
    if 1:        
        import pyagg
        c=pyagg.Canvas(1000,200)
        c.percent_space()
        x = 2
        for i,g in enumerate(diffgroups.keys()):
            print i, g
            x += 2
            c.draw_line([(x,0),(x,100)], fillcolor=g, fillsize=2)
        c.get_image().show()

    if 1:
        # view colors in image
        #im.show()
        quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256)
        quant.show()
        qcounts,qcolors = zip(*sorted(quant.getcolors(256), key=lambda x: x[0]))
        counts,colors = zip(*sorted(quant.convert('RGB').getcolors(256), key=lambda x: x[0]))
        qarr = np.array(quant)
##        for k,g in diffgroups_dict.items():
##            colim = im_arr.copy()
##            
##    ##        colim[np.isin(qarr, g, invert=True)] = [255,255,255] #[0,0,0]
##            
##            diffs = [pairdiffs[k,oth] for oth in qcolors]
##            difftable = dict(list(zip(qcolors,diffs)))
##            diff_im_flat = np.array(quant).flatten()
##            for qcol,diff in difftable.items():
##                diff_im_flat[diff_im_flat==qcol] = diff
##            diff_im = diff_im_flat.reshape((im.height, im.width))
##            dissim = diff_im > 10
##            colim[dissim] = (255,255,255)
##
##            PIL.Image.fromarray(colim).show()
        colim = np.zeros((im_arr.shape[0]*im_arr.shape[1]*3,), np.uint8).reshape(im_arr.shape)
        for k,g in diffgroups.items():
            qg = [qcolors[colors.index(gs)] for gs in g]
            mask = np.isin(qarr, qg)
            colim[mask] = k
        PIL.Image.fromarray(colim).show()

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
    
    print diff_im.min(), diff_im.mean(), diff_im.max()
    #quant.show()
    #PIL.Image.fromarray(diff_im).show()

    diff_im = diff_im / diff_im.max()
    #PIL.Image.fromarray(diff_im).show()
    diff_im *= 255
    #PIL.Image.fromarray(diff_im).show()
    diff_im = diff_im.astype(np.uint8)
    #PIL.Image.fromarray(diff_im).show()
    return diff_im

def image_segments(im):
    colorthresh = 5

    # BOX APPROACH

##    boxes = detect_boxes(diff_im)

    # WHITE THRESH APPROACH
    # TODO: Problem when entire maps is largely gray/white, maybe instead do largest area of high stdev??
    white = (255,255,255)
    im_prep = im
    im_prep = quantize(im_prep)
    thresh,mask = threshold(im_prep, white, colorthresh)
    mask_arr = np.zeros(mask.shape) #(im.size[0],im.size[1]))
    mask_arr = mask_arr.astype(np.uint8)
    mask_arr[mask] = 255
    #print thresh.size,mask_arr.shape
    #PIL.Image.fromarray(mask_arr).show()

    # detect contours
    print mask_arr.dtype
    contours,_ = cv2.findContours(mask_arr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # draw them
    #im_arr_draw = np.array(im)
    #cv2.drawContours(im_arr_draw, contours, -1, (0,255,0), 1)
    #PIL.Image.fromarray(im_arr_draw).show()

    # map is largest nonwhite contour
    largest_nonwhite = sorted(contours, key=lambda cnt: cv2.contourArea(cnt))[-1]
    map_outline = largest_nonwhite
    #map_mask = np.zeros(mask.shape)
    #cv2.drawContours(map_mask, [largest_nonwhite], -1, 255, -1)

    # TODO: Problem when map is irregular shape and text overflows into margins ala brazil
    # solution: expand borders in those cases??
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    #map_mask = cv2.dilate(map_mask,kernel,iterations=30)
    
    #map_im = np.array(im.convert('RGBA'))
    #map_im[map_mask==0] = 0
    #map_im = PIL.Image.fromarray(map_im)
    #map_im.show()

    # margins is the opposite
    #margins_mask = (np.ones(mask.shape) * 255) - map_mask
    #margins_im = np.array(im.convert('RGBA'))
    #margins_im[margins_mask==0] = 0
    #margins_im = PIL.Image.fromarray(margins_im)
    #margins_im.show()

    # detect boxes
    diff_im = color_edges(im)
    boxes = detect_boxes(diff_im)

    return map_outline, boxes

def mask_image(im, poly, invert=False):
    mask = np.zeros((im.size[1], im.size[0]))
    cv2.drawContours(mask, [poly], -1, 255, -1)
    new_im = np.array(im.convert('RGBA'))
    trueval = 255 if invert else 0
    new_im[mask==trueval] = 0
    new_im = PIL.Image.fromarray(new_im)
    return new_im

def mask_text(data, poly, invert=False):
    from shapely.geometry import Polygon, box
    poly = Polygon([tuple(p[0]) for p in poly])
    poly = poly.simplify(0)
    xmin,ymin,xmax,ymax = poly.bounds

    def maskfunc(p,b):
        return p.disjoint(b) if invert else p.intersects(b)

    def is_outside(r):
        return r['left'] > xmax or (r['left']+r['width']) < xmin or r['top'] > ymax or (r['top']+r['height']) < ymin

    def skipfunc(r):
        outside = is_outside(r)
        if invert: return not outside
        else: return outside

    for r in data:
        if skipfunc(r):
            continue
        else:
            textbox = box(r['left'], r['top'], r['left']+r['width'], r['top']+r['height'] )
            if maskfunc(poly, textbox):
                yield r

def detect_boxes(im):
    # detect boxes from contours
    # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
    # https://stackoverflow.com/questions/46641101/opencv-detecting-a-circle-using-findcontours
    # see esp: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    import numpy as np
    import cv2
    im_arr = np.array(im)
    thresh = 127 # from grayscale to binary (arbitrary, experiment more)
    im_arr[im_arr < thresh] = 0 
    im_arr[im_arr >= thresh] = 255
    #im_arr = cv2.cvtColor(im_arr, cv2.COLOR_RGB2GRAY)
    #PIL.Image.fromarray(im).show()
    contours,_ = cv2.findContours(im_arr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #im_arr_draw = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2RGB)
    boxes = []
    im_area = im_arr.shape[0] * im_arr.shape[1]
    for cnt in contours:
        epsilon = 5 # pixels # 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            if (im_area/(2.0**2)) > cv2.contourArea(cnt) > (im_area/(16.0**2)):
                boxes.append(approx)
            
    #cv2.drawContours(im_arr_draw, boxes, -1, (0,255,0), 1)
    #PIL.Image.fromarray(im_arr_draw).show()
    return boxes

def detect_data(im, bbox=None):
    if bbox:
        im = im.crop(bbox)
    data = t.image_to_data(im, lang='eng+fra', config='--psm 11') # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) for row in drows]
    return drows

def process_text(data, textconf):
    processed = []
    for r in data:
        r['function'] = None
        r['conf'] = float(r['conf'])
        if r['conf'] < textconf: continue
        # text
        r['text'] = r.get('text')
        if not r['text']: continue
        r['text_clean'] = re.sub('^\\W+|\\W+$', '', r['text'], flags=re.UNICODE) # strips nonalpha chars from start/end
        if not r['text_clean']: continue
        r['numeric'] = r['text_clean'].replace(' ','').isnumeric()
        r['uppercase'] = r['text_clean'].isupper()
        # coords
        for k in 'top left width height'.split():
            r[k] = int(r[k])
        # font
        r['fontheight'] = r['height']
        processed.append(r)

    return processed

def extract_metadata(data):
    metadata = {}
    
    # find map title
    metadata['title'] = sorted(data, key=lambda r: -r['fontheight'])[0]

    return metadata 

def connect_text(data, ythresh=6, xthresh=6):
    
    def merge_textgroups(newdata):
        for i in range(len(newdata)):
            group = newdata[i]
            dct = {'text': ' '.join([r['text'] for r in group]),
                   'text_clean': ' '.join([r['text_clean'] for r in group]),
                   'numeric': min([r['numeric'] for r in group]),
                   'uppercase': min([r['uppercase'] for r in group]),
                   'conf': sum([r['conf'] for r in group]) / float(len(group)),
                   'left': min([r['left'] for r in group]),
                   'top': min([r['top'] for r in group]),
                   'fontheight': max([r['fontheight'] for r in group]),
                   'function': group[0]['function'],
                   }
            dct['width'] = max([r['left']+r['width'] for r in group]) - dct['left']
            dct['height'] = max([r['top']+r['height'] for r in group]) - dct['top']
            #print len(group),dct
            newdata[i] = dct
        return newdata
    
    # connect texts horizontally
    candidates = sorted(data, key=lambda r: r['left'])
    newdata = []
    while candidates:
        r = candidates.pop(0)
        # find all whose top and bottom are within threshold
        totheright = []
        for r2 in candidates:
            if r2 == r: continue
            # height difference can't be more than x2
            if (max(r['height'],r2['height']) / float(min(r['height'],r2['height']))) > 2: 
                continue
            # top or bottom within threshold
            if (abs(r['top'] - r2['top']) < ythresh) or (abs((r['top']+r['height']) - (r2['top']+r2['height'])) < ythresh): 
                totheright.append(r2)
        # group those within height x distance
        group = [r]
        right = r['left'] + r['width']
        while totheright:
            nxt = totheright.pop(0)
            if (right + r['height']) > nxt['left']:
                # within distance, add to group
                right = nxt['left'] + nxt['width']
                candidates.pop(candidates.index(nxt)) # remove as candidate for others
                group.append(nxt)
            else:
                # not within distance, break loop
                break
        newdata.append(group)

    # merge groups
    newdata = merge_textgroups(newdata)

    # do same vertically (center aligned only)
    candidates = sorted(newdata, key=lambda r: r['top'])
    newdata = []
    while candidates:
        r = candidates.pop(0)
        # find all whose midpoints are within threshold
        below = []
        for r2 in candidates:
            if r2 == r: continue
            # height difference can't be more than x2
            if (max(r['height'],r2['height']) / float(min(r['height'],r2['height']))) > 2: 
                continue
            # midpoints within threshold
            mid1 = r['left'] + (r['width'] / 2.0)
            mid2 = r2['left'] + (r2['width'] / 2.0)
            if abs(mid1 - mid2) < xthresh:
                below.append(r2)
        # group those within height y distance
        group = [r]
        bottom = r['top'] + r['height']
        while below:
            nxt = below.pop(0)
            #print '---'
            #print bottom + r['height'], nxt['top']
            #print r
            #print nxt
            if (bottom + r['height']) > nxt['top']:
                # within distance, add to group
                bottom = nxt['top'] + nxt['height']
                candidates.pop(candidates.index(nxt)) # remove as candidate for others
                group.append(nxt)
            else:
                # not within distance, break loop
                break
        newdata.append(group)

    # merge groups
    newdata = merge_textgroups(newdata)

##    # merge text groups vertically (center aligned only)
##    candidates = sorted(newdata, key=lambda gr: min([r['top'] for r in gr]))
##    newdata2 = []
##    while candidates:
##        gr = candidates.pop(0)
##        below = []
##        # find all exactly below, ie whose midpoint is within threshold
##        for gr2 in candidates:
##            if gr2 == gr: continue
##            mid1 = (gr[0]['left'] + (gr[-1]['left'] + gr[-1]['width']) / 2.0)
##            mid2 = (gr2[0]['left'] + (gr2[-1]['left'] + gr2[-1]['width']) / 2.0)
##            if abs(mid1 - mid2) < xthresh:
##                # same midpoint
##                below.append(gr2)
##        # group those witihin height y distance
##        bottom = max([r['top']+r['height'] for r in gr])
##        grheight = bottom - min([r['top'] for r in gr])
##        while below:
##            nxtgr = below.pop(0)
##            if (bottom + grheight) > min([r['top'] for r in nxtgr]):
##                # within height distance, add to group
##                bottom = max([r['top']+r['height'] for r in nxtgr])
##                candidates.pop(candidates.index(nxtgr))
##                gr.extend(nxtgr)
##            else:
##                # not within distance, break loop
##                break
##        newdata2.append(gr)   

    return newdata

def detect_text_points(im, data, debug=False):
    # simple
##    points = []
##    for r in data:
##        text = r['text']
##        text = text.strip().strip(''' *'".,:''')
##        x = int(r['left'])
##        y = int(r['top'])
##        pt = (text, (x,y))
##        print pt
##        points.append( pt )
    

    # experiments...
##    import numpy as np
##    import cv2
##    points = []
##    im_arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
##    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
##    for r in data:
##        text = r['text']
##        text = text.strip().replace('.', '') #.replace("\x91", "'")
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##
##        # possibly, narrow down the actual bbox via avg pixel coords
####        onxs,onys = np.nonzero(im_arr == 0)
####        tot = len(onxs)
####        onxcounts = zip(*onxs.unique(return_counts=True))
####        onycounts = zip(*onys.unique(return_counts=True))
####        # ...hmm...
##
##        buff = int(h * 1.5)
##        filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff] = False # look in buffer zone around text box
##        filt_im_arr[y1:y2, x1:x2] = True # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
##    im_arr[filt_im_arr] = 255
##    im_arr[im_arr < 255] = 0
##    #PIL.Image.fromarray(im_arr).show()
##
##    # next detect shapes in the masked area
##    # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
##    # https://stackoverflow.com/questions/46641101/opencv-detecting-a-circle-using-findcontours
##    # see esp: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
##    _,contours,_ = cv2.findContours(im_arr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
##    im_arr_draw = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2RGB)
##    circles = []
##    centers = []
##    for cnt in contours:
##        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
##        area = cv2.contourArea(cnt)
##        # filled circles only
##        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
##        circleArea = radius * radius * np.pi
##        if (len(approx) > 8) and area >= 5 and 0.6 < circleArea/float(area) < 1.4:
##            circles.append(cnt)
##            centers.append((cx,cy))
##    cv2.drawContours(im_arr_draw, circles, -1, (0,255,0), 1)
##    PIL.Image.fromarray(im_arr_draw).show()
##
##    # link each text to closest point
##    import shapely, shapely.geometry
##    points = []
##    centers = [shapely.geometry.Polygon([tuple(cp[0]) for cp in c]) for c in circles]
##    for r in data:
##        text = r['text']
##        text = text.strip().replace('.', '') #.replace("\x91", "'")
##        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
##        x2 = x1+w
##        y2 = y1+h
##        rect = shapely.geometry.box(x1, y1, x2, y2)
##        # first those within dist of bbox
##        nearby = filter(lambda x: x[1] < h, [(c,rect.distance(c)) for c in centers])
##
##        # choose the nearest circle
##        if nearby:
##            nearest = sorted(nearby, key=lambda x: x[1])[0]
##            c = nearest[0]
##            print text,c
##            p = (int(c.centroid.x), int(c.centroid.y))
##            points.append((text, p))
##        
##        # or choose the one with most whitespace around
####        neighbourhoods = [(c, im_arr[int(c.centroid.y)-h:int(c.centroid.y)+h, int(c.centroid.x)-h:int(c.centroid.x)+h].sum())
####                          for c,_ in nearby]
####        loneliest = sorted(neighbourhoods, key=lambda x: -x[1])
####        if loneliest:
####            c = loneliest[0][0]
####            print text,c
####            p = (int(c.centroid.x), int(c.centroid.y))
####            points.append((text, p))


    # final?
    im_arr = np.array(im) #cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    #PIL.Image.fromarray(im_arr).show()
    
    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
    for r in data:
        #if not r['placename']:
        #    continue
        x1,y1,w,h = [r[k] for k in 'left top width height'.split()]
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
    import pyagg
    c = pyagg.canvas.from_image(PIL.Image.fromarray(dist_arr*50))
    
    points = []
    for r in data:
        #if not r['placename']:
        #    continue
        x1,y1,w,h = [r[k] for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        # buffer
        x1 -= h
        x2 += h
        y1 -= h
        y2 += h
        # first those within buffered bbox
        nearby = filter(lambda(x,y): x1 < x < x2 and y1 < y < y2, centers)

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

            for n in nearby:
                c.draw_circle(xy=n, fillsize=centervals[centers.index(n)], fillcolor=None, outlinecolor=(255,0,0))

            c.draw_circle(xy=maxpt, fillsize=1, fillcolor=None, outlinecolor=(0,0,255))

    if debug:
        c.get_image().show()
    
    return data

def triang(test, matchcandidates=None):
    # TODO: maybe superfluous, maybe just integrate right into "triangulate"?? 
    names,positions = zip(*test)
    # reverse ys due to flipped image coordsys
    maxy = max((y for x,y in positions))
    positions = [(x,maxy-y) for x,y in positions]
    # triangulate
    #print 99,names,positions
    matches = triangulate(names, positions, matchcandidates)
    #for f,diff,diffs in matches[:1]:
        #print 'error:', round(diff,6)
        #for c in f['properties']['combination']:
        #    print c
        #viewmatch(positions, f)
    return matches

def find_matches(test, thresh=0.1, minpoints=8, mintrials=8, maxiter=500, maxcandidates=10, n_combi=3, source='gns', debug=False):
    # filter to those that can be geocoded
    print 'geocode and filter'
    coder = geocode.OptimizedCoder()

    if source == 'best' or source == 'avg':
        mintrials = 30
        maxiter = 10000
    
    import time
    testres = []
    for nxtname,nxtpos in test:
        print 'geocoding',nxtname
        try:
            res = list(coder.geocode(nxtname, maxcandidates))
            if res:
                if source == 'avg':
                    import math
                    resnew = []
                    for r in res:
                        mayberes = [r2 for r2 in res if r != r2 and r['properties']['data'] != r2['properties']['data']]
                        x,y = r['geometry']['coordinates']
                        mayberes = [r2 for r2 in mayberes if math.hypot(abs(r2['geometry']['coordinates'][0]-x), abs(r2['geometry']['coordinates'][1]-y)) < 0.5]
                        if mayberes:
                            xs,ys = zip(*[r2['geometry']['coordinates'] for r2 in mayberes])
                            xm = sum(xs)/float(len(xs))
                            ym = sum(ys)/float(len(ys))
                            r['geometry'] = dict(type='Point', coordinates=(xm,ym))
                        resnew.append(r)
                    res = resnew
                elif source == 'best':
                    pass # just keep all the results and choose best matching ones
                else:
                    res = [r for r in res if r['properties']['data']==source]
                testres.append((nxtname,nxtpos,res))
                #time.sleep(0.1)
        except Exception as err:
            print 'EXCEPTION:', err
        
    #testres = [(nxtname,nxtpos,res)
    #           for nxtname,nxtpos,res in testres if res and len(res)<10]
    #testres = [(nxtname,nxtpos,res[:maxcandidates])
    #           for nxtname,nxtpos,res in testres]

    # print names to be tested
    for nxtname,nxtpos,res in testres:
        print nxtname,len(res)

    # find all triangles from all possible combinations        
    combis = itertools.combinations(testres, n_combi)
    # sort randomly to avoid local minima
    from random import uniform
    combis = sorted(combis, key=lambda x: uniform(0,1))
    # sort by length of possible geocodings, ie try most unique first --> faster+accurate
    combis = sorted(combis, key=lambda gr: sum((len(res) for nxtname,nxtpos,res in gr)))

    print '\n'+'finding all possible triangles of {} possible combinations'.format(len(combis))
    triangles = []
    for i,tri in enumerate(combis):
        if debug:
            print '-----'
            print 'try triangle %s of %s' % (i, len(combis))
            print '\n'.join([repr((tr[0],len(tr[2]))) for tr in tri])
        # try triang
        best = None
        try: best = triang([tr[:2] for tr in tri],
                           matchcandidates=[tr[2] for tr in tri])
        except Exception as err: print 'EXCEPTION RAISED:',err
        if best:
            f,diff,diffs = best[0]
            #print f
            if debug:
                print 'error:', round(diff,6)
            if diff < thresh:
                if debug:
                    print 'TRIANGLE FOUND'
                valid = [tr[:2] for tr in tri]

                # ...
                for nxtname,nxtpos,res in testres:
                    if debug:
                        print 'trying to add incrementally:',nxtname,nxtpos
                    orignames,origcoords = zip(*valid)
                    orignames,origcoords = list(orignames),list(origcoords)
                    matchnames = list(f['properties']['combination'])
                    matchcoords = list(f['geometry']['coordinates'][0])
                    
                    if nxtpos in origcoords: continue
                    maxy = max((y for x,y in origcoords))
                    maxy = max(maxy,nxtpos[1])
                    nxtposflip = (nxtpos[0],maxy-nxtpos[1])
                    origcoordsflip = [(x,maxy-y) for x,y in origcoords + [nxtpos]]
                    best = triangulate_add(zip(orignames,origcoordsflip),
                                           zip(matchnames,matchcoords),
                                           (nxtname,nxtposflip),
                                           res)
                    if not best: continue
                    mf,mdiff,mdiffs = best[0]
                    if mdiff < thresh:
                        if debug:
                            print 'ADDING'
                        valid.append((nxtname,nxtpos))
                        f = mf
                
                print '\n'+'MATCHES FOUND (error=%r)' % round(diff,6)
                print '>>>', repr([n for n,p in valid]),'-->',[n[:15] for n in f['properties']['combination']]

                triangles.append((valid,f,diff))
                
        if debug:
            print '%s triangles so far:' % len(triangles)
        
        #print '\n>>>'.join([repr((round(tr[2],6),[n for n,p in tr[0]],'-->',[n[:15] for n in tr[1]['properties']['combination']]))
        #                 for tr in triangles])
        
        if len(triangles) >= mintrials and max((len(v) for v,f,d in triangles)) >= minpoints:
            break

        if i >= maxiter:
            break

    # of all the trial triangles, choose only the one with lowest diff and longest chain of points
    triangles = sorted(triangles, key=lambda(v,f,d): (d,-len(v)) )
    orignames,origcoords = [],[]
    matchnames,matchcoords = [],[]
    print '\n'+'Final matchset:'
    for tri,f,diff in triangles[:1]: # only the first best triangle is used
        for (n,c),(mn,mc) in zip(tri, zip(f['properties']['combination'], f['geometry']['coordinates'][0])):
            print 'final',n,c,mn,mc
            if c in origcoords or mc in matchcoords: continue
            orignames.append(n)
            origcoords.append(c)
            matchnames.append(mn)
            matchcoords.append(mc)

    print 'final diff', diff
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)

def optimal_warp_order(tiepoints):
    # due to automation and high likelihood of errors, we set higher point threshold for polynomial order
    # compare to gdal: https://github.com/naturalatlas/node-gdal/blob/master/deps/libgdal/gdal/alg/gdal_crs.c#L186
    if len(tiepoints) >= 20:
        warp_order = 3
    elif len(tiepoints) >= 10:
        warp_order = 2
    else:
        warp_order = 1
    return warp_order

##def optimal_warp_order(tiepoints):
##    frompoints, topoints = zip(*tiepoints)
##    rmses = [polynomial(order, frompoints, topoints)[3]
##             for order in range(1,3+1)]
##    print rmses
##    order = rmses.index(min(rmses)) + 1
##    return order

def warp(im, outpath, tiepoints, order=None):
    import os
    print 'control points:', tiepoints

    gdal_trans_in = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.tif')
    gdal_trans_out = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.tif')
    im.save(gdal_trans_in)
    
    gcptext = ' '.join('-gcp {0} {1} {2} {3}'.format(imgx,imgy,geox,geoy) for (imgx,imgy),(geox,geoy) in tiepoints)
    call = 'gdal_translate -of GTiff {gcptext} "{gdal_trans_in}" "{gdal_trans_out}"'.format(gcptext=gcptext, gdal_trans_in=gdal_trans_in, gdal_trans_out=gdal_trans_out)
    os.system(call) 
    
    opts = '-r bilinear -co COMPRESS=NONE -dstalpha -overwrite'
    if order:
        if order == 'tps': opts += ' -tps'
        else: opts += ' -order {}'.format(order)
    call = 'gdalwarp {options} "{gdal_trans_out}" "{outpath}"'.format(options=opts, gdal_trans_out=gdal_trans_out, outpath=outpath)
    os.system(call)

    os.remove(gdal_trans_in)
    os.remove(gdal_trans_out)

def final_controlpoints(tiepoints, residuals, origs, matches, outpath=False):
    controlpoints = []
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    for (oc,mc),res in zip(tiepoints, residuals):
        i = origcoords.index(oc)
        oc,on,mc,mn = [lst[i] for lst in (origcoords,orignames,matchcoords,matchnames)]
        controlpoints.append([on,oc,mn,mc,res])

    if outpath:
        vec = pg.VectorData(fields='origname origx origy matchname matchx matchy residual'.split())
        for on,(ox,oy),mn,(mx,my),res in controlpoints:
            vec.add_feature([on,ox,oy,mn,mx,my,res], geometry={'type': 'Point', 'coordinates': (mx,my)})
        vec.save(outpath)

    return controlpoints

def debug_prep(im, outpath):
    im.save(outpath)

def debug_ocr(im, outpath, data, controlpoints, origs):
    import pyagg
    c = pyagg.canvas.from_image(im)
    #print c.width,c.height,im.size
    for r in data:
        top,left,w,h = [r[k] for k in 'top left width height'.split()]
        box = [left, top, left+w, top+h]
        text = r.get('text_clean','[?]')
        #print box,text
        c.draw_box(bbox=box, fillcolor=None, outlinecolor=(0,255,0))
        c.draw_text(text, xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0)) #bbox=box)
    for oname,ocoord in origs:
        c.draw_circle(xy=ocoord, fillsize=1, fillcolor=None, outlinecolor=(0,0,255))
    for on,oc,mn,mc,res in controlpoints:
        c.draw_circle(xy=oc, fillsize=1, fillcolor=(255,0,0,155), outlinecolor=None)
    c.save(outpath)

def debug_warped(pth, outpath, controlpoints):
    import pythongis as pg
    m = pg.renderer.Map(width=2000, height=2000, background='white')

    m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp",
                fillcolor=(217,156,38))

    warped = pg.RasterData(pth)
    for b in warped.bands:
        b.nodataval = 0 # need better approach, use 4th band as mask
    rlyr = m.add_layer(warped, transparency=0.3)

    m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_populated_places_simple\ne_10m_populated_places_simple.shp",
                fillcolor='red', fillsize=0.1) #outlinewidth=0.1)

    anchors = pg.VectorData(fields=['origname', 'matchname', 'residual'])
    for on,oc,mn,mc,res in controlpoints:
        anchors.add_feature([on,mn,res], dict(type='Point', coordinates=mc))
    m.add_layer(anchors, fillcolor=(0,255,0), fillsize=0.3)

    m.zoom_bbox(*rlyr.bbox)
    m.zoom_out(1.5)
    #m.view()
    m.save(outpath)
    

def automap(inpath, outpath=None, matchthresh=0.1, textcolor=None, colorthresh=25, textconf=60, bbox=None, warp_order=None, max_residual=0.05, **kwargs):
    start = time.time()
    
    print 'loading image', inpath
    im = PIL.Image.open(inpath).convert('RGB')
    if bbox:
        im = im.crop(bbox)

    # determine various paths
    infold,infil = os.path.split(inpath)
    infil,ext = os.path.splitext(infil)
    if not outpath:
        outpath = os.path.join(infold, infil + '_georeferenced.tif')
    outfold,outfil = os.path.split(outpath)

    # partition image
    mapp_poly,box_polys = image_segments(im)
    #im.show()

    # begin prep
    im_prep = im

    # detect map box
    #boxes = detect_boxes(im_prep)
    #sefsdf

    # histogram testing
    #print 'detecting colors'
    #colorgroups = maincolors(im_prep)

    # NOTE: upscale -> threshold creates phenomenal resolution and ocr detections, but is slower and demands much more memory
    # consider doing threshold -> upscale if memoryerror...

    # upscale for better ocr
    print 'upscaling'
    im_prep = im_prep.resize((im_prep.size[0]*2, im_prep.size[1]*2), PIL.Image.LANCZOS)

    # precalc color differences (MAYBE move to inside threshold?)
    print 'quantize'
    im_prep = quantize(im_prep)
    #counts,colors = zip(*im_prep.getcolors(256))
    #pairdiffs = color_differences(colors)

    # for each color
    data = []

##    print 'test black...'
##    im_prep_thresh = threshold(im_prep, (0,0,0), colorthresh)
##    im_prep_thresh.show()
##    print 'test black ocr...'
##    print len(detect_data(im_prep_thresh))

    # ALTERNATIVELY, resize to 4 times size, detect text of any color
    # based on color change boundaries/standard deviation
    # ...

    # placenames should be mostly in grayscale colors, so limit ocr to different levels of grayish colors
    # only loop detected colors that are similar to various grayshades
    #textcolors = [(v,v,v) for v in range(0, 150+1, 50)] # grayshades
    textcolors = [textcolor] # input color
    for color in textcolors: 

        # threshold
        print 'thresholding', color
        im_prep_thresh,mask = threshold(im_prep, color, colorthresh)
        debugpath = os.path.join(outfold, infil+'_debug_prep.png')
        debug_prep(im_prep_thresh, debugpath)

        # ocr
        print 'detecting text'
        subdata = detect_data(im_prep_thresh)
        subdata = process_text(subdata, textconf)

        # assign text characteristics
        for dct in subdata:
            dct['color'] = color

        # filter out duplicates from previous loop, in case includes some of the same pixels
        subdata = [r for r in subdata
                   if (r['top'],r['left'],r['width'],r['height'])
                   not in [(dr['top'],dr['left'],dr['width'],dr['height']) for dr in data]
                   ]

        # connect text data
        subdata = connect_text(subdata)

        # detect text coordinates
        print 'determening text anchors'
        subdata = detect_text_points(im_prep_thresh, subdata)

        data.extend(subdata)
        print 'text data size', len(subdata), len(data)

    # downscale the data coordinates of the upscaled image back to original coordinates
    for r in data: 
        for k in 'top left width height'.split():
            r[k] = int(r[k]) / 2
        # same for points
        if 'anchor' in r:
            x,y = r['anchor']
            r['anchor'] = (x/2, y/2)

    # print map metadata
    print 'map metadata:'
    metadata = extract_metadata(data)
    for k,v in metadata.items():
        print k
        print v

    # end prep, switch back to original image
    im = im

    # decide which labels to consider placenames
    # only labels inside map region
    filt = mask_text(data, mapp_poly)
    # excluding labels inside any boxes
    for box in box_polys:
        filt = mask_text(filt, box, invert=True)
    # only nonnumeric, first uppercased, rest lowercased
    filt = [r for r in filt
            if r['numeric'] is False and r['text_clean'][0].isupper() and not r['uppercase']]
    # mark as placename
    for r in filt:
        r['function'] = 'placename'
    # final placenames with anchor points
    points = [(r['text_clean'], r['anchor']) for r in data if r['function']=='placename' and 'anchor' in r]

    # find matches
    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)
    
    print 'finding matches'

    # (experimental)
##    for src in 'avg un natearth ciesin osm geonames gns'.split() + [None]:
##        print src
##        try:
##            origs,matches = find_matches(points, matchthresh, source=src, **kwargs)
##            orignames,origcoords = zip(*origs)
##            matchnames,matchcoords = zip(*matches)
##            tiepoints = zip(origcoords, matchcoords)
##            warp_order_auto = optimal_warp_order(tiepoints)
##            frompoints,topoints = zip(*tiepoints)
##            best, best_frompoints, best_topoints, best_residuals = optimal_rmse(warp_order or warp_order_auto, frompoints, topoints, max_residual=max_residual)
##            print 'RMSE:', best
##        except Exception as err:
##            print err
##    fdsfds
        
    origs,matches = find_matches(points, matchthresh, source='gns', **kwargs)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)

    # determine transform method
    if not warp_order:
        warp_order_auto = optimal_warp_order(tiepoints)

    # exclude outliers
    if warp_order == 'tps':
        best_residuals = [None for _ in tiepoints]
    else:
        print '\n'+'excluding outliers'
        frompoints,topoints = zip(*tiepoints)
        best, best_frompoints, best_topoints, best_residuals = optimal_rmse(warp_order or warp_order_auto, frompoints, topoints, max_residual=max_residual)
        tiepoints = zip(best_frompoints, best_topoints)
        print 'RMSE:', best

    # once again, determine transform method after excluding outliers
    if not warp_order:
        warp_order_auto = optimal_warp_order(tiepoints)

    # warp
    print '\n'+'warping'
    print '{} points, warp_method={}'.format(len(tiepoints), warp_order or warp_order_auto)
    mapp_im = mask_image(im, mapp_poly)
    warp(mapp_im, outpath, tiepoints, warp_order or warp_order_auto)

    # final control points
    cppath = os.path.join(outfold, infil+'_controlpoints.geojson')
    controlpoints = final_controlpoints(tiepoints, best_residuals, origs, matches, outpath=cppath)

    # draw data onto image
    debugpath = os.path.join(outfold, infil+'_debug_ocr.png')
    debug_ocr(im, debugpath, data, controlpoints, origs)

    # view warpedp
    print '\n'+'finished!'
    print 'total runtime: {:.1f} seconds \n'.format(time.time() - start)
    debugpath = os.path.join(outfold, infil+'_debug_warp.png')
    debug_warped(outpath, debugpath, controlpoints)

def drawpoints(img):
    import pythongis as pg
    from pythongis.app import dialogs, icons

    import tk2

    points = []
    
    class ClickControl(tk2.basics.Label):
        def __init__(self, master, *args, **kwargs):
            tk2.basics.Label.__init__(self, master, *args, **kwargs)

            icon = os.path.abspath("automap/resources/flag.png")
            self.clickbut = tk2.basics.Button(self, command=self.begin_click)
            self.clickbut.set_icon(icon, width=40, height=40)
            self.clickbut.pack()

            self.mouseicon_tk = icons.get(icon, width=30, height=30)

        def begin_click(self):
            print "begin click..."
            # replace mouse with identicon
            self.mouseicon_on_canvas = self.mapview.create_image(-100, -100, anchor="center", image=self.mouseicon_tk )
            #self.mapview.config(cursor="none")
            def follow_mouse(event):
                # gets called for entire app, so check to see if directly on canvas widget
                root = self.winfo_toplevel()
                rootxy = root.winfo_pointerxy()
                mousewidget = root.winfo_containing(*rootxy)
                if mousewidget == self.mapview:
                    curx,cury = self.mapview.canvasx(event.x) + 28, self.mapview.canvasy(event.y) + 5
                    self.mapview.coords(self.mouseicon_on_canvas, curx, cury)
            self.followbind = self.winfo_toplevel().bind('<Motion>', follow_mouse, '+')
            # identify once clicked
            def callclick(event):
                # reset
                cancel()
                # find
                x,y = self.mapview.mouse2coords(event.x, event.y)
                self.click(x, y)
            self.clickbind = self.winfo_toplevel().bind("<ButtonRelease-1>", callclick, "+")
            # cancel with esc button
            def cancel(event=None):
                self.winfo_toplevel().unbind('<Motion>', self.followbind)
                self.winfo_toplevel().unbind('<ButtonRelease-1>', self.clickbind)
                self.winfo_toplevel().unbind('<Escape>', self.cancelbind)
                #self.mapview.config(cursor="arrow")
                self.mapview.delete(self.mouseicon_on_canvas)
            self.cancelbind = self.winfo_toplevel().bind("<Escape>", cancel, "+")

        def click(self, x, y):
            print "clicked: ",x, y
            entrywin = tk2.Window()
            entrywin.focus()

            title = tk2.Label(entrywin, text="Coordinates: %s, %s" % (x, y))
            title.pack(fill="x")#, expand=1)

            entry = tk2.Entry(entrywin, label="Place Name: ", width=40)
            entry.pack(fill="x", expand=1)
            entry.focus()

            def addpoint():
                name = entry.get()
                print x,y,name

                coder = geocode.OptimizedCoder()
                ms = coder.geocode(name, limit=10)
                if ms:
                    points.append((name, (x,y)))
                    
                    markers.add_feature([name], {'type':'Point', 'coordinates':(x,y)})
                    markerslyr.update()
                    self.mapview.threaded_rendering()
                    #self.mapview.renderer.render_all() #render_one(markers)
                    #self.mapview.renderer.update_draworder()
                    #self.mapview.update_image()
                else:
                    tk2.messagebox.showwarning('Try another named location!', 'Named location "%s" could not be found/geocoded.' % name)
                
                entrywin.destroy()
            
            okbut = tk2.OkButton(entrywin, command=addpoint)
            okbut.pack()

    # load image
    import PIL, PIL.Image
    r = pg.RasterData('testmaps/testorig.jpg',
                      xy_cell=(0,0),
                      xy_geo=(0,0),
                      cellwidth=1,
                      cellheight=1,
                      width=img.size[0],
                      height=img.size[1],
                      )
    m = pg.renderer.Map()
    m._create_drawer()
    m.drawer.custom_space(*r.bbox)
    m.add_layer(r)

    markers = pg.VectorData(fields=['name'])
    markerslyr = m.add_layer(markers, fillcolor=(0,255,0), fillsize=1)
    markerslyr.add_effect('shadow', xdist=5, ydist=5) # opacity=0.8
    #markerslyr.add_effect('glow', color='black', size=20)

    m.zoom_auto()

    # build and run application
    w = tk2.Tk()
    w.state('zoomed')
    mw = pg.app.builder.MultiLayerMap(w, m)
    mw.pack(fill="both", expand=1)
    clickcontrol = ClickControl(mw.mapview)
    clickcontrol.place(relx=0.99, rely=0.98, anchor="se")
    mw.mapview.add_control(clickcontrol)
    w.mainloop()

    print points
    return points

def manual(pth, matchthresh=0.1, bbox=None, warp_order=None, **kwargs):
    print 'loading image'
    im = PIL.Image.open(pth).convert('RGB')
    if bbox:
        im = im.crop(bbox)

    # determine various paths
    infold,infil = os.path.split(inpath)
    infil,ext = os.path.splitext(infil)
    if not outpath:
        outpath = os.path.join(infold, infil + '_georeferenced.tif')
    outfold,outfil = os.path.split(outpath)

    # manually select text anchors/coordinates
    print 'drawing anchor points'
    points = drawpoints(im)

    # find matches
    print 'finding matches'
    origs,matches = find_matches(points, matchthresh, **kwargs)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    print tiepoints
    for on,mc,mn in zip(orignames,matchcoords,matchnames):
        print on,mc,mn

    # determine transform method
    if not warp_order:
        warp_order = optimal_warp_order(tiepoints)

    # warp
    print 'warping'
    warp(im, outpath, tiepoints, warp_order)

    # final control points
    cppath = os.path.join(outfold, infil+'_controlpoints.geojson')
    controlpoints = final_controlpoints(tiepoints, best_residuals, origs, matches, outpath=cppath)

    # view warped
    debug_warped(outpath, controlpoints)

    

    
