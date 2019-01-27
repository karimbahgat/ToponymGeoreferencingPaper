
from automap import triangulate, normalize, geocode, triangulate_add

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
    print 99,names,positions
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

def process_optim(test, thresh=0.1, limit=10):
    # build three random triangulations and choose best
    from random import uniform
    randidx = sorted(range(len(test)), key=lambda x: uniform(0,1))

    def trytriang():
        # choose 3 that are valid
        import random
        valid = []
        while len(valid) < 3:
            nxtname,nxtpos = random.choice(test)
            if (nxtname,nxtpos) not in valid and geocode(nxtname):
                valid.append((nxtname,nxtpos))
            
        # try triang
        best = triang(valid)
        if best:
            f,diff,diffs = best[0]
            print f
            if diff < thresh:
                return tuple(valid),diff,f
    
    trials = []
    while len(trials) < 3:
        tr = trytriang()
        if tr:
            trials.append(tr)
            
    print 'hmm'
    for tr in trials: print tr
    
    valid,diff,f = sorted(trials, key=lambda x: x[1])

    # drop chosen ones from list
    print 'hmm2'
    for v in valid:
        print v
##        test.pop(test.index(v))

    print valid
    print f

    # any remaining places are added incrementally to existing triangle/shape
    orignames,origcoords = list(zip(*valid))
    orignames,origcoords = list(orignames),list(origcoords)
    matchnames = list(f['properties']['combination'])
    matchcoords = list(f['geometry']['coordinates'][0])

    #maxy = max((y for x,y in origcoords))
    #origcoordsflip = [(x,maxy-y) for x,y in origcoords]
    #viewmatch(origcoordsflip, f)
    
    while test:
        print '-----'
        nxtname,nxtpos = test.pop(0)
        maxy = max((y for x,y in origcoords))
        maxy = max(maxy,nxtpos[1])
        nxtposflip = (nxtpos[0],maxy-nxtpos[1])
        origcoordsflip = [(x,maxy-y) for x,y in origcoords + [nxtpos]]
        print nxtname,nxtpos
        prevdiff = diff
        best = triangulate_add(zip(orignames,origcoordsflip), zip(matchnames,matchcoords), (nxtname,nxtposflip))
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


if __name__ == '__main__':
    pth = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #pth = 'testmaps/indo_china_1886.jpg'
    #pth = 'testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg'
    #pth = 'testmaps/2113087.jpg'
    im = PIL.Image.open(pth).crop((0,0,1500,1500))

    # threshold
    thresh = 60
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
    im.show()
    data = detect_data(im)
    data = filter(lambda r: r['text'] and len(r['text'].strip().replace(' ','')) >= 3 and int(r['conf']) > 60,
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

##    points = '''(u'ur.', (408, 106))
##                (u'Lum', (273, 131))
##                (u'Bet', (163, 382))
##                (u'Nir', (204, 383))
##                (u'Gal\u2018on', (44, 461))
##                (u'Zanmr', (343, 550))
##                (u'\\allm\u2018h', (339, 571))
##                (u'Bet', (222, 597))
##                (u'Guvrin', (262, 597))
##                (u'lamp!', (207, 645))
##                (u'(iuvrin', (273, 646))
##                (u'zldzixh', (0, 757))
##                (u'Lakhlsh', (42, 908))
##                (u'\xbbAmazya', (369, 1081))
##                (u'Ben', (304, 1524))
##                (u'Mrstma', (350, 1525))
##                (u"Nl'dlt", (1127, 0))
##                (u'Zekha', (546, 11))
##                (u'Zumet', (615, 82))
##                (u'11411:', (701, 82))
##                (u'"ZurHadassa', (1299, 5))
##                (u'Husan', (1591, 27))
##                (u'Betar', (1382, 84))
##                (u"'IHit", (1457, 84))
##                (u'"Khat', (1714, 86))
##                (u'gamer', (875, 112))
##                (u'Netiy', (652, 139))
##                (u'Lamed', (745, 140))
##                (u'"\u201c\u2018\u201d\u2026', (872, 130))
##                (u'lama)\u201d', (637, 167))
##                (u'Nahh\xe4l\xeen', (1502, 179))
##                (u'[amer', (718, 196))
##                (u'_:r', (865, 203))
##                (u'AVIWBF', (1018, 194))
##                (u'Nahal', (1285, 202))
##                (u'Geva\u2018ot', (1352, 202))
##                (u'Giv\u2018at', (495, 216))
##                (u'Yesh\u2018yahu', (560, 217))
##                (u'\u201c\u201d\u201c/"""', (708, 214))
##                (u'New\xe9', (1633, 225))
##                (u'Damyy\xe9l', (1697, 226))
##                (u'oNeweMikh\u2018\xe8\xee', (913, 253))
##                (u'gape.', (1200, 244))
##                (u'Rosh', (1523, 270))
##                (u'S\xfcrlm', (1583, 261))
##                (u'\u2018Zafrinm', (595, 311))
##                (u'Adderel', (762, 310))
##                (u'AHon', (1519, 327))
##                (u'azar', (1641, 314))
##                (u'\u2018Shevu\xee', (1494, 349))
##                (u'OEfrata', (1689, 341))
##                (u'Bat', (1339, 370))
##                (u'\u2018Aym\u2019', (1380, 371))
##                (u'401,12!', (1603, 393))
##                (u'S\xdcF\xcf\xce', (1150, 402))
##                (u"(\xeem'h", (1590, 408))
##                (u'\u2018Egvon', (1646, 409))
##                (u'KefarEgyon', (1412, 424))
##                (u'Mtgdal', (1639, 478))
##                (u'\u201807', (1712, 479))
##                (u'Ben', (1631, 521))
##                (u'Fajj\xe9r', (1678, 522))
##                (u'BeitUmma-', (1290, 567))
##                (u"'Mu", (1597, 568))
##                (u'askare\u2018Am', (1651, 573))
##                (u'oth\xe0r\xe4s', (1090, 593))
##                (u'Karm\xe9', (1347, 630))
##                (u'Zur', (1420, 631))
##                (u'.N\xfcba', (1061, 652))
##                (u'BeitAm\xe9', (1048, 709))
##                (u"Si'Tria", (1566, 775))
##                (u'Tarqumiya', (810, 811))
##                (u'Ben', (1204, 846))
##                (u'K\xe9\u2018nd', (1250, 846))
##                (u'Halhm', (1427, 845))
##                (u'Shuvukh', (1612, 842))
##                (u'Teiem', (1007, 889))
##                (u'ldna', (759, 946))
##                (u'oAdora', (925, 974))
##                (u"\u201c.\ufb01'h", (1089, 1051))
##                (u'Sumaiya', (639, 1080))
##                (u'DeirS\xe0mit', (724, 1109))
##                (u'\xabSheqef', (559, 1148))
##                (u"\u2018A'", (630, 1196))
##                (u'D\xfcr\xe4', (973, 1179))
##                (u"'Bani!", (1704, 1183))
##                (u'en!', (599, 1197))
##                (u'wwe', (649, 1201))
##                (u'(Adorz\xe0yrm)', (941, 1206))
##                (u'S\\kka', (559, 1290))
##                (u'NahaJ', (801, 1291))
##                (u'Negohot', (867, 1292))
##                (u'Haqga', (1274, 1349))
##                (u'Arr', (1261, 1389))
##                (u'Pen\xe9', (1645, 1384))
##                (u'Hever', (1704, 1385))
##                (u'MEN', (562, 1389))
##                (u'Kharasa', (903, 1399))
##                (u'aka.', (1243, 1393))
##                (u'Rihw', (1282, 1463))'''.split('\n')
##    points = '''(u'\xabr\u2026\u2026', (0, 0))
##                (u'Bet', (163, 382))
##                (u'Niro', (204, 383))
##                (u'Gal\u2018on', (44, 461))
##                (u'lama!', (343, 550))
##                (u'Valmvh', (339, 571))
##                (u'BetGuvrin', (222, 597))
##                (u'lol\u2026)!', (207, 645))
##                (u'(iuvrin', (273, 646))
##                (u':\u201d\u2026', (0, 757))
##                (u'Lakhish', (42, 908))
##                (u'\\Amazya', (368, 1081))
##                (u'Ben', (304, 1524))
##                (u'Mirstm.', (349, 1524))
##                (u"lvldu'", (1127, 0))
##                (u'Zekha', (546, 11))
##                (u'vaet', (615, 82))
##                (u"zu'qa", (701, 82))
##                (u'1mm!', (894, 115))
##                (u'Netiv', (652, 139))
##                (u'Lamed', (745, 140))
##                (u'\u201d\u2026\u2026"', (872, 130))
##                (u'Zur', (1316, 5))
##                (u'Hadassa', (1358, 6))
##                (u'!.,', (1422, 60))
##                (u'B\xe9tar', (1382, 68))
##                (u'\u2018Illit', (1457, 84))
##                (u'R\xeeh\xeeya', (1281, 1463))
##                (u'H\xfcs\xe4n', (1591, 27))
##                (u'z,\u2026', (637, 170))
##                (u"Na'hh\xe4l\xefn", (1502, 167))
##                (u'Eh!', (674, 191))
##                (u'.amct', (726, 200))
##                (u'AVI\u2018\xcbZSf', (1018, 194))
##                (u'Nahal', (1285, 202))
##                (u'Geva\u2018ot', (1352, 202))
##                (u'Giv\u2018atYesh\u2018yahu', (495, 216))
##                (u'.Newe\u2018', (1612, 225))
##                (u'Damyy\xe9l', (1697, 226))
##                (u'News', (931, 252))
##                (u'Mikh\u2018e\u2019i', (995, 253))
##                (u'Rosh', (1523, 270))
##                (u'sim\u2026', (1582, 260))
##                (u'Ei\u2018azar', (1611, 308))
##                (u"'Zafrlnm", (595, 311))
##                (u'Adderet', (762, 310))
##                (u'Allen', (1520, 327))
##                (u"'Shevut", (1494, 349))
##                (u'OEfrata', (1689, 341))
##                (u'(107%', (1590, 390))
##                (u"l\\'l", (1610, 414))
##                (u"'UH", (1680, 414))
##                (u"Kefar'Ezyon", (1412, 415))
##                (u'Migdal', (1639, 478))
##                (u'\u2018Oz', (1712, 478))
##                (u'BeitFaji\xe4r', (1631, 521))
##                (u'Bei\xeeUmmao', (1290, 567))
##                (u"'Mu\u2018askareV\u2018Arr", (1597, 568))
##                (u'\xabKh\xe4r\xe4s', (1090, 593))
##                (u'Karm\xe9', (1347, 630))
##                (u'Zur', (1420, 631))
##                (u"'NDba", (1061, 652))
##                (u'BeitAul\xe4', (1048, 709))
##                (u'Si\u2018\xcfr\u2018', (1566, 775))
##                (u'Tarqumiya', (810, 811))
##                (u'Ben', (1204, 846))
##                (u'Kenn', (1250, 846))
##                (u'Halb\xfcl', (1427, 845))
##                (u'Shuyukh', (1612, 842))
##                (u'Telem', (1006, 889))
##                (u'!dna', (759, 945))
##                (u',Adora', (921, 974))
##                (u'T.\ufb01\xffh', (1089, 1051))
##                (u'Sumaiya', (639, 1080))
##                (u"Qn\u2018yatArba'", (1425, 1090))
##                (u'DeirS\xe4mit', (724, 1109))
##                (u'(;\u2014', (1452, 1120))
##                (u'.Sheqef', (559, 1148))
##                (u'Hevron', (1379, 1166))
##                (u'Dura', (973, 1179))
##                (u"'Bam", (1704, 1183))
##                (u'Ben.\u2018Awwa', (584, 1195))
##                (u'(Adorayim)', (941, 1206))
##                (u'(Hebron)', (1378, 1196))
##                (u'Sikka', (559, 1290))
##                (u'jf".', (735, 1290))
##                (u'NahaiNegohot', (801, 1291))
##                (u'Haggay', (1274, 1349))
##                (u'Pen\xe9', (1644, 1384))
##                (u'Hever', (1704, 1385))
##                (u'Mald', (562, 1389))
##                (u'Kharasa', (903, 1399))
##                (u'Fawar', (1230, 1388))'''.split('\n')
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
        c.draw_text(text, xy=(left,top), anchor='sw', textsize=6) #bbox=box)
    c.save('testmaps/testocr.png')

    # process and warp
    origs,matches = process_optim(points, 0.1)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    print tiepoints



    
