
from automap import triangulate, normalize, geocode, triangulate_add

import itertools

import pytesseract as t

import PIL, PIL.Image


def detect_data(im, bbox=None):
    if bbox:
        im = im.crop(bbox)
    data = t.image_to_data(im, lang='eng+fra') # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) for row in drows]
    return drows

def triang(test, matchcandidates=None):
    names,positions = zip(*test)
    # reverse ys due to flipped image coordsys
    maxy = max((y for x,y in positions))
    positions = [(x,maxy-y) for x,y in positions]
    # triangulate
    print 99,names,positions
    matches = triangulate(names, positions, matchcandidates)
    for f,diff,diffs in matches[:1]:
        print 'error:', round(diff,6)
        for c in f['properties']['combination']:
            print c
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

def process_optim(test, thresh=0.1, limit=10):
    # filter to those that can be geocoded
    print 'geocode and filter'
    testres = [(nxtname,nxtpos,list(geocode(nxtname)))
               for nxtname,nxtpos in test]
    testres = [(nxtname,nxtpos,res)
               for nxtname,nxtpos,res in testres if res]

    # sort by length of possible geocodings, ie try most unique first --> faster+accurate
    testsort = sorted(testres, key=lambda(nxtname,nxtpos,res): len(res))
    for nxtname,nxtpos,res in testsort:
        print nxtname,len(res)

    # find initial triangle from all possible combinations
    valid = []
    for tri in itertools.combinations(testsort, 3):
        print 'try triangle'
        for tr in tri:
            print tr[0],len(tr[2])
        # try triang
        try: best = triang([tr[:2] for tr in tri],
                           matchcandidates=[tr[2] for tr in tri])
        except Exception as err: print 'EXCEPTION RAISED:',err
        if best:
            f,diff,diffs = best[0]
            print f
            if diff < thresh:
                valid = [tr[:2] for tr in tri]
                break

    print 'found initial triangle'
    for v in valid:
        print v
        testsort.pop([tr[:2] for tr in tri].index(v))

    print '===',f

    # any remaining places are added incrementally to existing triangle/shape
    orignames,origcoords = list(zip(*valid))
    orignames,origcoords = list(orignames),list(origcoords)
    matchnames = list(f['properties']['combination'])
    matchcoords = list(f['geometry']['coordinates'][0])

    #maxy = max((y for x,y in origcoords))
    #origcoordsflip = [(x,maxy-y) for x,y in origcoords]
    #viewmatch(origcoordsflip, f)

    print 'adding to triangle incrementally'
    for nxtname,nxtpos,res in testsort:
        print '-----'
        maxy = max((y for x,y in origcoords))
        maxy = max(maxy,nxtpos[1])
        nxtposflip = (nxtpos[0],maxy-nxtpos[1])
        origcoordsflip = [(x,maxy-y) for x,y in origcoords + [nxtpos]]
        print nxtname,nxtpos
        prevdiff = diff
        best = triangulate_add(zip(orignames,origcoordsflip),
                               zip(matchnames,matchcoords),
                               (nxtname,nxtposflip),
                               res)
        if not best:
            continue
        f,diff,diffs = best[0]
        print f
        print 'error:', round(diff,6)
        
        #viewmatch(origcoordsflip, f)
        
        if diff < thresh: # and diff/prevdiff < 10: # stay within thresh and dont worsen more than 10x
            print 'ADDING'
            orignames.append(nxtname)
            origcoords.append(nxtpos)
            matchnames = f['properties']['combination']
            matchcoords = f['geometry']['coordinates'][0]

        if len(matchnames) >= limit:
            break
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)

def warp(image, tiepoints):
    import os
    print 'control points:', tiepoints
    gcptext = ' '.join('-gcp {0} {1} {2} {3}'.format(imgx,imgy,geox,geoy) for (imgx,imgy),(geox,geoy) in tiepoints)
    call = 'gdal_translate -of GTiff {gcptext} "{image}" "testmaps/warped.tif" & pause'.format(gcptext=gcptext, image=image)
    os.system(call)
    os.system('gdalwarp -r bilinear -tps -co COMPRESS=NONE -dstalpha -overwrite "testmaps/warped.tif" "testmaps/warped2.tif" & pause')



if __name__ == '__main__':
    #pth = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #pth = 'testmaps/indo_china_1886.jpg'
    #pth = 'testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg'
    #pth = 'testmaps/2113087.jpg'
    pth = 'testmaps/burkina.jpg'
    #pth = 'testmaps/cameroon_pol98.jpg'
    im = PIL.Image.open(pth) #.crop((0,0,2000,2000))

    # threshold
    thresh = 50
    target = (0,0,0)
    px = im.load()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            rgb = px[x,y]
            diff = abs( sum([target[i]-rgb[i] for i in range(3)])/3.0 )
            if diff < thresh:
                #print rgb,diff
                pass #px[x,y] = target
            else:
                px[x,y] = (255,255,255)

    # ocr
    im.save('testmaps/testthresh.jpg')
    data = detect_data(im)
    data = filter(lambda r: r.get('text') and len(r['text'].strip().replace(' ','')) >= 3 and int(r['conf']) > 60,
                  data)
    points = []
    for r in data:
        text = r['text']
        text = text.strip().replace('.', '') #.replace("\x91", "'")
        x = int(r['left'])
        y = int(r['top'])
        pt = (text, (x,y))
        print pt
        points.append( pt )

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
    c = pyagg.load(pth)
    for r in data:
        top,left,w,h = [int(r[k]) for k in 'top left width height'.split()]
        box = [left, top, left+w, top+h]
        text = r.get('text','[?]')
        print box,text
        c.draw_box(bbox=box, fillcolor=None, outlinecolor=(0,255,0))
        c.draw_text(text, xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0)) #bbox=box)
    c.save('testmaps/testocr.png')

    # process and warp
    origs,matches = process_optim(points, 0.1)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    print tiepoints
    for on,mc,mn in zip(orignames,matchcoords,matchnames):
        print on,mc,mn
    warp(pth, tiepoints)

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


    

    
