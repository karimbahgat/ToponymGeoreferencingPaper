
import itertools
import math

import numpy as np

import PIL
from PIL import Image, ImageCms, ImageMorph, ImageFilter, ImageOps, ImageMath

import cv2

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff_matrix import delta_e_cie2000


def rgb_to_lab(im):
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    im = ImageCms.applyTransform(im, rgb2lab)
    return im

def color_difference(im, color):
    im_arr = np.array(im)

    # If wanna do larger more quicker, then do it tiled, eg:
##    tw,th = 300,300
##    diff = np.zeros(tuple(reversed(im.size)))
##    for y in range(0,im.size[1],th):
##        print y
##        for x in range(0,im.size[0],tw):
##            x2 = min(x + tw, im.size[0])
##            y2 = min(y + th, im.size[1])
##            tile = im.crop([x,y,x2,y2])
##            tdiff = mapfit.segmentation.color_difference(tile, col)
##            tdiff[tdiff>thresh] = 255
##            diff[y:y2, x:x2] = tdiff
##    Image.fromarray(diff).show()

    #target = color
    #diffs = [pairdiffs[tuple(sorted([target,oth]))] for oth in colors if oth != target]

    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
    #counts,colors = zip(*im.getcolors(256))
    colors,counts = np.unique(im_arr.reshape(-1, 3), return_counts=True, axis=0)
    colors,counts = map(tuple,colors), map(int,counts)
    colors_lab = [convert_color(sRGBColor(*col, is_upscaled=True), LabColor).get_value_tuple()
                  for col in colors]
    diffs = delta_e_cie2000(target, np.array(colors_lab))
    
    difftable = dict(list(zip(colors,diffs)))
    diff_im = np.zeros((im.height,im.width))
    im_arr_idx = im_arr.dot(np.array([1, 256, 65536])) #im_arr[:,:,0] + (im_arr[:,:,1]*256) + (im_arr[:,:,2]*256*256)
    for col,diff in difftable.items():
        #print col
        
        #diff_im[(im_arr[:,:,0]==col[0])&(im_arr[:,:,1]==col[1])&(im_arr[:,:,2]==col[2])] = diff # 3 lookup is slow

        #wherecolor = (im_arr==col).all(axis=2) 
        #diff_im[wherecolor] = diff

        wherecolor = im_arr_idx == (col[0] + (col[1]*256) + (col[2]*256*256))
        diff_im[wherecolor] = diff

    return diff_im

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

def group_colors(colors, thresh=10, categories=None):
    pairdiffs = color_differences(colors)
    diffgroups = dict([(c,[]) for c in categories]) if categories else dict()
    for c in colors:
        # find closest existing group that is sufficiently similar
        dists = [(gc,pairdiffs[tuple(sorted([c,gc]))])
                 for gc in diffgroups.keys()]
        similar = [(gc,dist) for gc,dist in dists if c != gc and dist < thresh]
        if similar:
            # find the most similar group color
            nearest = sorted(similar, key=lambda x: x[1])[0][0]
            diffgroups[nearest].append(c) 
            # update that group key as the new central color (lowest avg dist to group members)
            gdists = [(gc1,[pairdiffs[tuple(sorted([c,gc]))] for gc2 in diffgroups[nearest]]) for gc1 in diffgroups[nearest]]
            central = sorted(gdists, key=lambda(gc,gds): sum(gds)/float(len(gds)))[0][0]
            diffgroups[central] = diffgroups.pop(nearest)
        else:
            # create new group
            diffgroups[c] = [c]
            
    # set each group key to avg color of group members
##    for c,gcols in diffgroups.items():
##        rs,gs,bs = zip(*gcols)
##        avgcol = np.mean(rs), np.mean(gs), np.mean(bs)
##        diffgroups[avgcol] = diffgroups.pop(c)
            
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

def quantize(im):
    quant = im.convert('P', palette=PIL.Image.ADAPTIVE, colors=256).convert('RGB')
    return quant

def mask_image(im, poly, invert=False):
    mask = np.zeros((im.size[1], im.size[0]))
    cv2.drawContours(mask, [poly], -1, 255, -1)
    #PIL.Image.fromarray(mask).show()
    new_im = np.array(im)
    trueval = 255 if invert else 0
    new_im[mask==trueval] = 0
    new_im = PIL.Image.fromarray(new_im)
    return new_im

def edge_filter(im):
    # preprocess
    #im = im.filter(ImageFilter.BoxBlur(3))
    #im = im.filter(ImageFilter.EDGE_ENHANCE())
    #im.show()
    
    # lab values
    lab = rgb_to_lab(im)
    l,a,b = lab.split()

    # find edges of avg
##    avg = (np.array(l) + np.array(a) + np.array(b)) / 3.0
##    avg = PIL.Image.fromarray(avg.astype(np.uint8))
##    edges = avg.filter(ImageFilter.FIND_EDGES())

    # find max edge
    edges_l = l.filter(ImageFilter.FIND_EDGES())
    edges_a = a.filter(ImageFilter.FIND_EDGES())
    edges_b = b.filter(ImageFilter.FIND_EDGES())
    edges = ImageMath.eval('convert(max(max(l,a),b), "L")', {'l':edges_l, 'a':edges_a, 'b':edges_b})

    # threshold
    edges = edges.point(lambda v: 255 if v >= 255/2 else 0)
    #edges.show()
    
    # remove outer edge? 
    edges = ImageOps.crop(edges, 1)
    edges = ImageOps.expand(edges, 1, fill=0)
    return edges

def close_edge_gaps(im):
    grow = ImageMorph.MorphOp(op_name='dilation4')
    changes,im = grow.apply(im)
    #changes,im = grow.apply(im)
    
    shrink = ImageMorph.MorphOp(op_name='erosion4')
    changes,im = shrink.apply(im)
    #changes,im = shrink.apply(im)

    return im

def detect_map_outline(im):
    # assumes image shows grayscale edges
    edges = im
    
    # find contours
    contours,_ = cv2.findContours(np.array(edges), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # find map frame as contour larger than third of the image
    largest = sorted(contours, key=lambda cnt: cv2.contourArea(cnt))[-1]
    w,h = im.size
    im_area = w*h
    if cv2.contourArea(largest) > im_area/(3.0**2):
        # maybe also require that contour doesnt touch edges of the image
        xbuf,ybuf = 4,4 #w/100.0, h/100.0 # 1 percent
        if largest[:,:,0].min() > 0+xbuf and largest[:,:,0].max() < w-xbuf \
           and largest[:,:,1].max() > 0+ybuf and largest[:,:,1].max() < h-ybuf:
            return largest

def detect_boxes(im):
    # detect boxes from contours
    # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
    # https://stackoverflow.com/questions/46641101/opencv-detecting-a-circle-using-findcontours
    # see esp: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    contours,_ = cv2.findContours(np.array(im), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #im_arr_draw = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2RGB)
    boxes = []
    im_area = im.size[0] * im.size[1]
    for cnt in contours:
        epsilon = 5 # pixels # 0.01*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            if (im_area/(2.0**2)) > cv2.contourArea(cnt) > (im_area/(16.0**2)):
                boxes.append(approx)
            
    #cv2.drawContours(im_arr_draw, boxes, -1, (0,255,0), 1)
    #PIL.Image.fromarray(im_arr_draw).show()
    return boxes

##class Quad:
##    def __init__(self, x, y, w, h):
##        self.x = x
##        self.y = y
##        self.w = w
##        self.h = h
##        self.children = []
##
##    def __repr__(self):
##        return 'Quad({}, {}, {}, {}, children={})'.format(self.x, self.y, self.w, self.h, len(self.children))
##
##    def bbox(self):
##        x1,y1 = self.x - self.w/2.0, self.y - self.h/2.0
##        x2,y2 = x1+self.w, y1+self.h
##        return x1,y1,x2,y2
##        
##    def split(self):
##        if self.children:
##            for subq in self.children:
##                subq.split()
##        else:
##            halfwidth = self.w/2.0
##            halfheight = self.h/2.0
##            quad = Quad(self.x-halfwidth/2.0, self.y-halfheight/2.0,
##                        halfwidth, halfheight)
##            self.children.append(quad)
##            quad = Quad(self.x+halfwidth/2.0, self.y-halfheight/2.0,
##                        halfwidth, halfheight)
##            self.children.append(quad)
##            quad = Quad(self.x+halfwidth/2.0, self.y+halfheight/2.0,
##                        halfwidth, halfheight)
##            self.children.append(quad)
##            quad = Quad(self.x-halfwidth/2.0, self.y+halfheight/2.0,
##                        halfwidth, halfheight)
##            self.children.append(quad)
##
##    def sample(self):
##        if self.children:
##            q = self.children.pop(0)
##            if q.children:
##                # if child quad has children
##                subq = q.sample() # sample the quad
##                self.children.append(q) # send to back of the line
##                return subq
##            else:
##                return q

class Quad:
    def __init__(self, x, y, w, h):
        '''Breadth-first quad tree that infinitely samples each quad to get a spatially balanced sampling with finer and finer detail,
        and splits subquads when necessary'''
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.children = []

    def __repr__(self):
        return 'Quad({}, {}, {}, {}, children={})'.format(self.x, self.y, self.w, self.h, len(self.children))

    def bbox(self):
        x1,y1 = self.x - self.w/2.0, self.y - self.h/2.0
        x2,y2 = x1+self.w, y1+self.h
        return x1,y1,x2,y2
        
    def split(self):
        if self.children:
            # already has children, use existing
            pass
        else:
            halfwidth = self.w/2.0
            halfheight = self.h/2.0
            # upper left
            quad = Quad(self.x-halfwidth/2.0, self.y-halfheight/2.0,
                        halfwidth, halfheight)
            self.children.append(quad)
            # lower right
            quad = Quad(self.x+halfwidth/2.0, self.y+halfheight/2.0,
                        halfwidth, halfheight)
            self.children.append(quad)
            # upper right
            quad = Quad(self.x+halfwidth/2.0, self.y-halfheight/2.0,
                        halfwidth, halfheight)
            self.children.append(quad)
            # lower left
            quad = Quad(self.x-halfwidth/2.0, self.y+halfheight/2.0,
                        halfwidth, halfheight)
            self.children.append(quad)

    def sample(self):
        '''Samples this quad, either as itself or by further sampling the next in line of its child quads.'''
        # if doesnt have any children, split and return self 
        if not self.children:
            self.split()
            return self
        # get next quad
        q = self.children.pop(0)
        samp = q.sample()
        self.children.append(q) # send to back of the line
        return samp
        




#####################

def image_segments(im):
    '''This is the main user function'''
    # resize to smaller resolution
    w,h = im.size
    dw,dh = 1000,1000
    ratio = max(w/float(dw), h/float(dh))
    im_small = im.resize((int(w/ratio), int(h/ratio)), PIL.Image.ANTIALIAS)

    # filter to edges
    edges = edge_filter(im_small)

    # grow to close small edge holes
    edges = close_edge_gaps(edges)

    # NOTE: masking should be done on each sample image

    # find map outline
    map_outline = detect_map_outline(edges)
    if map_outline is not None:
        # convert back to original image coords
        map_outline = (map_outline * ratio).astype(np.uint64)

    # find boxes
    boxes = detect_boxes(edges)
    # convert back to original image coords
    boxes = [(box*ratio).astype(np.uint64) for box in boxes]

    return map_outline, boxes

def sample_quads(bbox, samplesize):
    '''Samples quads of samplesize from a bbox region in a rotating fashion, until all parts of bbox region has been sampled'''
    x1,y1,x2,y2 = bbox
    w = abs(x2-x1)
    h = abs(y2-y1)
    sw,sh = samplesize

    # divide image into quad tiles
    cx = (x1+x2)/2.0
    cy = (y1+y2)/2.0
    quads = Quad(cx, cy, w, h)
    quads.split()

    # sample quads in a rotating fashion until all parts of bbox has been covered by the samplesize
    #xmin,ymin,xmax,ymax = min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)
    #sxmin,symin,sxmax,symax = cx,cy,cx,cy
    intersects_centroid = 0
    while True:
        # sample 4 quads
        for _ in range(4):
            q = quads.sample()
            sq = Quad(q.x, q.y, sw, sh)
            yield sq

            # check if sample quad intersects the region centroid
            _sxmin,_symin,_sxmax,_symax = sq.bbox()
            if not (_sxmax < cx or _sxmin > cx or _symax < cy or _symin > cy):
                intersects_centroid += 1

        # when 4 sample quads touches the region centroid, the entire region has been covered
        if intersects_centroid >= 4:
            break


