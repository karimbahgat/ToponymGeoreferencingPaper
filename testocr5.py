
from automap import triangulate, normalize, geocode, triangulate_add

import itertools

import pytesseract as t

import PIL, PIL.Image


def threshold(im, color, thresh):
##    from colormath.color_objects import sRGBColor, LabColor
##    from colormath.color_conversions import convert_color
##    from colormath.color_diff import delta_e_cie2000
##    color = (0,0,0)
##    target = convert_color(sRGBColor(*color), LabColor)
##    px = im.load()
##    coldict = dict()
##    for y in range(im.size[1]):
##        for x in range(im.size[0]):
##            rgb = px[x,y]
##            if rgb in coldict:
##                diff = coldict.get(rgb)
##                #print 'getting',rgb
##            else:
##                pxcol = convert_color(sRGBColor(*[v/255.0 for v in rgb]), LabColor)
##                diff = delta_e_cie2000(target, pxcol)
##                #print 'adding',rgb
##                coldict[rgb] = diff
##            if diff < thresh:
##                #print rgb,diff
##                pass #px[x,y] = target
##            else:
##                px[x,y] = (255,255,255)

##    from colormath.color_objects import sRGBColor, LabColor
##    from colormath.color_conversions import convert_color
##
##    from PIL import ImageCms
##    srgb_profile = ImageCms.createProfile("sRGB")
##    lab_profile  = ImageCms.createProfile("LAB")
##    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
##    lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
##
##    color = (0,0,0)
##    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
##    tl,ta,tb = target
##    threshsq = thresh**2
##    px = im.load()
##    lab_px = lab_im.load()
##    coldict = dict()
##    for y in range(im.size[1]):
##        for x in range(im.size[0]):
##            lab = lab_px[x,y]
##            #print lab,target
##            if lab in coldict:
##                diff = coldict.get(lab)
##                #print 'getting',rgb
##            else:
##                l,a,b = lab
##                diff = (tl-l)**2 + (ta-a)**2 + (tb-b)**2
##                print diff,threshsq
##                #print 'adding',rgb
##                coldict[lab] = diff
##            if diff < threshsq:
##                #print rgb,diff
##                print diff,threshsq
##                pass #px[x,y] = target
##            else:
##                px[x,y] = (255,255,255)



##    from colormath.color_objects import sRGBColor, LabColor
##    from colormath.color_conversions import convert_color
##    from colormath.color_diff_matrix import delta_e_cie2000
##
##    import numpy as np
##
##    # PIL approach is not perfect, conversion to LAB is not fully accurate
##    # TODO: switch to this: https://code.i-harness.com/en/q/3142c9
##    # see https://stackoverflow.com/questions/21210479/converting-from-rgb-to-lab-colorspace-any-insight-into-the-range-of-lab-val
##    from PIL import ImageCms
##    srgb_profile = ImageCms.createProfile("sRGB")
##    lab_profile  = ImageCms.createProfile("LAB")
##    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
##    lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
##
##    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
##    lab_im_arr = np.array(lab_im).astype(np.float32) / 255.0 * 100 # PIL uses 0-255, but colormath expects 0-100
##    w,h,_ = lab_im_arr.shape
##    diff_im = delta_e_cie2000(target, lab_im_arr.flatten().reshape((w*h, 3))).reshape((w,h))
##    dissim = diff_im >= thresh
##
##    im_arr = np.array(im) 
##    im_arr[dissim] = (255,255,255)
##
##    im = PIL.Image.fromarray(im_arr)

    

##    from colormath.color_objects import sRGBColor, LabColor
##    from colormath.color_conversions import convert_color
##    from colormath.color_diff_matrix import delta_e_cie2000
##
##    import numpy as np
##
##    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor)
##
####    func = lambda rgb: convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor).get_value_tuple()
####    vfunc = np.vectorize(func)
####    im_arr = np.array(im)
####    lab_im_arr = vfunc(im_arr)
##
##    im_arr = np.array(im)
##    lab_im_arr = np.array(im)
##
##    coldict = dict()
##    for y in range(im.size[1]):
##        for x in range(im.size[0]):
##            rgb = tuple(im_arr[y,x])
##            if rgb in coldict:
##                lab = coldict.get(rgb)
##            else:
##                lab = convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor).get_value_tuple()
##            lab_im_arr[y,x] = lab
##
##    print im_arr
##    print lab_im_arr
##
##    fdsfs
##
##    w,h,_ = lab_im_arr.shape
##    diff_im = delta_e_cie2000(target, lab_im_arr.flatten().reshape((w*h, 3))).reshape((w,h))
##    dissim = diff_im >= thresh
##    
##    im_arr[dissim] = (255,255,255)
##
##    im = PIL.Image.fromarray(im_arr)

    # skimage approach
    from colormath.color_objects import sRGBColor, LabColor
    from colormath.color_conversions import convert_color
    from colormath.color_diff_matrix import delta_e_cie2000

    import numpy as np
    
    from skimage.color import rgb2lab
    
    im_arr = np.array(im)
    w,h,_ = im_arr.shape
    lab_im_arr = rgb2lab(im_arr)

    target = convert_color(sRGBColor(*color, is_upscaled=True), LabColor).get_value_tuple()
    w,h,_ = lab_im_arr.shape
    diff_im = delta_e_cie2000(target, lab_im_arr.flatten().reshape((w*h, 3))).reshape((w,h))
    dissim = diff_im >= thresh

    im_arr[dissim] = (255,255,255)

    im = PIL.Image.fromarray(im_arr)


    # TODO: Maybe also do gaussian or otsu binarization/smoothing?
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    

    return im

def maincolors(im, classes=5):
    import pyagg
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

    # using lab dist...
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
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # By Adrian Rosebrock
    import numpy as np
    import cv2

    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)
     
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
     
        # return the histogram
        return hist

    # Reshape the image to be a list of pixels
    im = im.resize((im.size[0]/50, im.size[1]/50))
    image_array = np.array(im).reshape((im.size[0] * im.size[1], 3))
    print len(image_array)

    def drawhist(bins):
        c = pyagg.Canvas(1200, 500)
        c.custom_space(0, 500, 1200, 0)
        maxval = max((b[0] for b in bins))
        x = 0
        bins = sorted(bins, key=lambda b: -b[0])[:1000]
        incr = c.width/float(len(bins))
        for cn,col in bins:
            c.draw_box(bbox=[x,0,x+incr,cn/float(maxval)*c.height], fillcolor=tuple(col))
            x += incr
        c.view()

    # Clusters the pixels
    clt = KMeans(n_clusters=classes)
    clt.fit(image_array)

    # Finds how many pixels are in each cluster
    hist = centroid_histogram(clt)

    bins = list(zip(hist, clt.cluster_centers_))
    for perc,col in bins:
        print perc,col

    drawhist(bins)

##    bestSilhouette = -1
##    bestClusters = 0
##
##    for clusters in range(3, 5): 
##        print 'clusters',clusters
##        
##        # Cluster colours
##        clt = KMeans(n_clusters = clusters)
##        clt.fit(image_array)
##
##        # Validate clustering result
##        silhouette = silhouette_score(image_array, clt.labels_, metric='euclidean')
##
##        # Find the best one
##        if silhouette > bestSilhouette:
##            bestSilhouette = silhouette
##            bestClusters = clusters

def detect_data(im, bbox=None):
    if bbox:
        im = im.crop(bbox)
    data = t.image_to_data(im, lang='eng+fra', config='--psm 11') # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) for row in drows]
    return drows

def detect_text_points(im, data):
    import numpy as np
    import cv2
    points = []
    im_arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    filt_im_arr = np.ones(im_arr.shape[:2], dtype=bool)
    for r in data:
        text = r['text']
        text = text.strip().replace('.', '') #.replace("\x91", "'")
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        buff = int(h * 1.5)
        filt_im_arr[y1-buff:y2+buff, x1-buff:x2+buff] = False # look in buffer zone around text box
        #filt_im_arr[y1:y2, x1:x2] = True # but do not look inside text region itself (NOTE: is sometimes too big and covers the point too)
    im_arr[filt_im_arr] = 255
    im_arr[im_arr < 255] = 0
    #PIL.Image.fromarray(im_arr).show()

    # next detect shapes in the masked area
    # https://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
    # https://stackoverflow.com/questions/46641101/opencv-detecting-a-circle-using-findcontours
    # see esp: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    _,contours,_ = cv2.findContours(im_arr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    im_arr_draw = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2RGB)
    circles = []
    centers = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        area = cv2.contourArea(cnt)
        # filled circles only
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        circleArea = radius * radius * np.pi
        if (len(approx) > 8) and area >= 5: #and 0.4 < circleArea/float(area) < 1.6:
            circles.append(cnt)
            centers.append((cx,cy))
    cv2.drawContours(im_arr_draw, circles, -1, (0,255,0), 1)
    PIL.Image.fromarray(im_arr_draw).show()

    # link each text to closest point
    import shapely, shapely.geometry
    points = []
    centers = [shapely.geometry.Polygon([tuple(cp[0]) for cp in c]) for c in circles]
    for r in data:
        text = r['text']
        text = text.strip().replace('.', '') #.replace("\x91", "'")
        x1,y1,w,h = [int(r[k]) for k in 'left top width height'.split()]
        x2 = x1+w
        y2 = y1+h
        rect = shapely.geometry.box(x1, y1, x2, y2)
        # first those within dist of bbox
        nearby = filter(lambda x: x[1] < h, [(c,rect.distance(c)) for c in centers])
        
        # then choose the one with most whitespace around
        neighbourhoods = [(c, im_arr[int(c.centroid.y)-h:int(c.centroid.y)+h, int(c.centroid.x)-h:int(c.centroid.x)+h].sum())
                          for c,_ in nearby]
        loneliest = sorted(neighbourhoods, key=lambda x: -x[1])
        if loneliest:
            c = loneliest[0][0]
            print text,c
            p = (int(c.centroid.x), int(c.centroid.y))
            points.append((text, p))

        # altern, narrow down the actual bbox via avg pixel coords
##        onxs,onys = np.nonzero(im_arr == 0)
##        tot = len(onxs)
##        onxcounts = zip(*onxs.unique(return_counts=True))
##        onycounts = zip(*onys.unique(return_counts=True))
##        # ...hmm...
    
    return points

def triang(test, matchcandidates=None):
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

def process(test, thresh=0.1):
    size = 3
    grouped = zip(*(iter(test),) * size) # grouped 3-wise
    grouped = [list(g) for g in grouped]
    remaind = len(test) % size
    for i in range(remaind):
        grouped[-i].append(test[-i])
    
    orignames = []
    origcoords = []
    matchnames = []
    matchcoords = []
    for group in grouped:
        print '-----'
        print group
        best = triang(group)[0]
        f,diff,diffs = best
        print f
        if diff < thresh:
            orignames.extend(zip(*group)[0])
            origcoords.extend(zip(*group)[1])
            matchnames.extend(f['properties']['combination'])
            matchcoords.extend(f['geometry']['coordinates'][0])
    return zip(orignames, origcoords), zip(matchnames, matchcoords)

def process_optim(test, thresh=0.1, minpoints=8, mintrials=8, maxiter=500, maxcandidates=10, n_combi=3):
    # filter to those that can be geocoded
    print 'geocode and filter'
    import time
    testres = []
    for nxtname,nxtpos in test:
        print 'geocoding',nxtname
        try:
            res = list(geocode(nxtname))
            if res:
                testres.append((nxtname,nxtpos,res))
                #time.sleep(0.1)
        except Exception as err:
            print 'EXCEPTION:', err
        
    #testres = [(nxtname,nxtpos,res)
    #           for nxtname,nxtpos,res in testres if res and len(res)<10]
    testres = [(nxtname,nxtpos,res[:maxcandidates])
               for nxtname,nxtpos,res in testres]

    # find all triangles from all possible combinations
    for nxtname,nxtpos,res in testres:
        print nxtname,len(res)
    combis = itertools.combinations(testres, n_combi)
    # sort randomly to avoid local minima
    from random import uniform
    combis = sorted(combis, key=lambda x: uniform(0,1))
    # sort by length of possible geocodings, ie try most unique first --> faster+accurate
    combis = sorted(combis, key=lambda gr: sum((len(res) for nxtname,nxtpos,res in gr)))

    print 'finding all possible triangles'
    triangles = []
    for i,tri in enumerate(combis):
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
            print 'error:', round(diff,6)
            if diff < thresh:
                print 'TRIANGLE FOUND'
                valid = [tr[:2] for tr in tri]

                # ...
                for nxtname,nxtpos,res in testres:
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
                        print 'ADDING'
                        valid.append((nxtname,nxtpos))
                        f = mf

                triangles.append((valid,f,diff))
                
        print '%s triangles so far:' % len(triangles)
        print '\n>>>'.join([repr((round(tr[2],6),[n for n,p in tr[0]],'-->',[n[:15] for n in tr[1]['properties']['combination']]))
                         for tr in triangles])
        if len(triangles) >= mintrials and max((len(v) for v,f,d in triangles)) >= minpoints:
            break

        if i >= maxiter:
            break

    # of all the trial triangles, choose only the one with longest chain of points and lowest diff
    triangles = sorted(triangles, key=lambda(v,f,d): (-len(v),-d) )
    orignames,origcoords = [],[]
    matchnames,matchcoords = [],[]
    for tri,f,diff in triangles[:1]: # only the first best triangle is used
        for (n,c),(mn,mc) in zip(tri, zip(f['properties']['combination'], f['geometry']['coordinates'][0])):
            print 'final',n,c,mn,mc
            if c in origcoords or mc in matchcoords: continue
            orignames.append(n)
            origcoords.append(c)
            matchnames.append(mn)
            matchcoords.append(mc)
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)

def warp(image, tiepoints):
    import os
    print 'control points:', tiepoints
    gcptext = ' '.join('-gcp {0} {1} {2} {3}'.format(imgx,imgy,geox,geoy) for (imgx,imgy),(geox,geoy) in tiepoints)
    call = 'gdal_translate -of GTiff {gcptext} "{image}" "testmaps/warped.tif"'.format(gcptext=gcptext, image=image)
    os.system(call) #-order 3 -refine_gcps 20 4
    os.system('gdalwarp -r bilinear -order 1 -co COMPRESS=NONE -dstalpha -overwrite "testmaps/warped.tif" "testmaps/warped2.tif"')



if __name__ == '__main__':
    #pth = 'testmaps/indo_china_1886.jpg'
    #pth = 'testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg'
    #pth = 'testmaps/2113087.jpg'
    #pth = 'testmaps/egypt_admn97.jpg'
    #pth = 'testmaps/brazil_army_amazon_1999.jpg'
    #pth = 'testmaps/brazil_pop_1977.jpg'
    #pth = 'testmaps/brazil_pol_1981.gif'
    #pth = 'testmaps/brazil_land_1977.jpg'
    
    pth = 'testmaps/burkina.jpg'
    #pth = 'testmaps/cameroon_pol98.jpg'
    #pth = 'testmaps/repcongo.png'
    #pth = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #pth = 'testmaps/egypt_pol_1979.jpg'
    #pth = 'testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg'
    #pth = 'testmaps/gmaps.png'
    im = PIL.Image.open(pth)#.crop((2000,2000,4000,4000))
    im = im.convert('RGB')
    im.save('testmaps/testorig.jpg')

    # histogram testing
    #maincolors(im)
    #fsdf

    # threshold
    im = threshold(im, (0,0,0), 25) # black for text, low
    #im = threshold(im, (0,0,0), 40) # black for text, high
    #im = threshold(im, (190,55,10), 20) # red 

    # ocr
    im.save('testmaps/testthresh.jpg')
    data = detect_data(im) #.resize((im.size[0]*2, im.size[1]*2), PIL.Image.ANTIALIAS)) # IF UPSCALE
    data = filter(lambda r: r.get('text') and len(r['text'].strip().replace(' ','')) >= 3 and int(r['conf']) > 60,
                  data)
    
    #for r in data: # IF UPSCALED
    #    for k in 'top left width height'.split():
    #        r[k] = int(r[k]) / 2
            
    points = detect_text_points(im, data)

##    points = '''(u'Zekharya', (546, 11))
##(u'Zur', (1308, 5))
##(u'Hadassa', (1358, 6))
##(u'Betar', (1382, 84))
##(u'Zomet', (615, 82))
##(u'Bet', (163, 382))
##(u'Nir', (204, 383))
##(u'Zomet', (343, 550))
##(u'Bet', (222, 597))
##(u'Guvrin', (262, 597))
##(u'Zomet', (207, 645))
##(u'Guyrin', (273, 646))
##(u'Lakhish', (42, 908))
##(u'LiOn', (555, 190))
##(u'Ela', (674, 191))
##(u'Zomer', (718, 196))
##(u'Givat', (495, 216))
##(u'Nahal', (1285, 202))
##(u"Geva'ot", (1352, 202))
##(u'Newe', (931, 241))
##(u"Jab'ae", (1215, 244))
##(u'Zattirim', (612, 311))
##(u'Adderet,', (762, 305))
##(u'Sanit', (1150, 402))
##(u'Beit', (1290, 567))
##(u'Ummae', (1337, 567))
##(u'Karm\xe9', (1347, 630))
##(u'Zur', (1420, 631))
##(u'N\xfcba', (1080, 657))
##(u'Beit', (1048, 709))
##(u'Aula', (1093, 709))
##(u'Tarqumiya', (810, 811))
##(u'Beit', (1204, 846))
##(u'K\xe4hil', (1250, 846))
##(u'Telem', (1006, 889))
##(u'eAdora', (921, 974))
##(u'Sumaiya', (639, 1080))
##(u'Deir', (724, 1109))
##(u'Samit', (773, 1109))
##(u'Sheqef', (577, 1148))
##(u'Durs', (973, 1179))
##(u'Hevron', (1379, 1166))
##(u'ura', (989, 1185))
##(u'Belt', (584, 1195))
##(u'(Adorayim)', (941, 1206))
##(u'(Hebron)', (1378, 1196))
##(u'Sikka', (559, 1290))
##(u'NahalNegohot', (801, 1291))
##(u'Haggay', (1274, 1349))
##(u'Majd', (562, 1389))
##(u'Kharasa', (903, 1399))
##(u'Faw\xe4r', (1230, 1388))'''.split('\n')
##    points = [eval(l.strip()) for l in points]

    # draw data onto image
    import pyagg
    c = pyagg.load('testmaps/testorig.jpg').resize(im.size[0]*2, im.size[1]*2)
    print c.width,c.height,im.size
    for r in data:
        top,left,w,h = [int(r[k]) for k in 'top left width height'.split()]
        box = [left, top, left+w, top+h]
        text = r.get('text','[?]')
        print box,text
        c.draw_box(bbox=box, fillcolor=None, outlinecolor=(0,255,0))
        c.draw_text(text, xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0)) #bbox=box)
    for txt,p in points:
        c.draw_circle(xy=p, fillsize=1, fillcolor=None, outlinecolor=(0,0,255))
    c.save('testmaps/testocr.png')

    # process and warp
    origs,matches = process_optim(points, 0.1)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    print tiepoints
    for on,mc,mn in zip(orignames,matchcoords,matchnames):
        print on,mc,mn
    warp('testmaps/testorig.jpg', tiepoints)

    # view warped
    import pythongis as pg
    m = pg.renderer.Map()

    m.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp")

    rlyr = m.add_layer('testmaps/warped2.tif')

    m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_populated_places_simple\ne_10m_populated_places_simple.shp",
                fillcolor='red', outlinewidth=0.1)
    
    anchors = pg.VectorData()
    for coord in matchcoords:
        anchors.add_feature([], dict(type='Point', coordinates=coord))
    m.add_layer(anchors, fillcolor=(0,255,0), outlinewidth=0.2)
    
    m.zoom_bbox(*rlyr.bbox)
    m.zoom_out(2)
    m.view()


    

    
