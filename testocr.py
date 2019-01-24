
from automap import triangulate, normalize

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

def triang(test):
    names,positions = zip(*test)
    # reverse ys due to flipped image coordsys
    maxy = max((y for x,y in positions))
    positions = [(x,maxy-y) for x,y in positions]
    # triangulate
    matches = triangulate(names, positions)
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


if __name__ == '__main__':
    pth = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #pth = 'testmaps/indo_china_1886.jpg'
    im = PIL.Image.open(pth)

    # threshold
    thresh = 100
    target = (0,0,0)
    px = im.load()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            rgb = px[x,y]
            diff = abs( sum([target[i]-rgb[i] for i in range(3)])/3.0 )
            if diff < thresh:
                #print rgb,diff
                px[x,y] = target
            else:
                px[x,y] = (255,255,255)

    # ocr
    im.show()
    data = detect_data(im)
    points = []
    for r in data:
        text = r.get('text', None)
        if text and len(text.strip().replace(' ','')) >= 3:
            x = int(r['left'])
            y = int(r['top'])
            pt = (text, (x,y))
            print pt
            points.append( pt )

    # draw data onto image
    import pyagg
    c = pyagg.load(pth)
    for r in data:
        top,left,w,h = [int(r[k]) for k in 'top left width height'.split()]
        box = [left, top, left+w, top+h]
        print box
        c.draw_box(bbox=box, fillcolor=None, outlinecolor=(0,255,0))
        #c.draw_text(r['text'], xy=(left,top)) #bbox=box)
    c.save('testmaps/testocr.png')

    # process and warp
    origs,matches = process(points)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    print tiepoints



    
