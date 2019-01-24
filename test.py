"""
"""

import os
from automap import triangulate, normalize, geocode, triangulate_add

def drawpoints(imgpath):
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

                import geopy
                coder = geopy.geocoders.Nominatim()
                ms = coder.geocode(name, exactly_one=False, limit=100)
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
    img = PIL.Image.open(imgpath)
    r = pg.RasterData(imgpath,
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

def viewmatch(positions, feat):
    import pythongis as pg

    m = pg.renderer.Map()

    d = pg.VectorData()
    geoj = {'type': 'Polygon',
            'coordinates': [list(positions)]
            }
    geoj['coordinates'][0].append(geoj['coordinates'][0][0]) # close poly
    geoj = normalize(geoj)
    d.add_feature([], geoj)
    m.add_layer(d, fillcolor='green', outlinecolor=None)

    d = pg.VectorData()
    geoj = {'type': 'Polygon',
            'coordinates': [ list(feat['geometry']['coordinates'][0]) ]
            }
    geoj['coordinates'][0].append(geoj['coordinates'][0][0]) # close poly
    geoj = normalize(geoj)
    print 'red',geoj
    d.add_feature([], geoj)
    m.add_layer(d, fillcolor=None, outlinecolor='red')

    m.zoom_auto()
    m.zoom_out(1.2)
    m.view()

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

def process_optim(test, thresh=0.1):
    # build initial triangulation from first 3
    valid = []
    while True:
        nxtname,nxtpos = test.pop(0)
        matches = geocode(nxtname)
        if matches:
            valid.append((nxtname,nxtpos))
        if len(valid) == 3:
            best = triang(valid)[0]
            f,diff,diffs = best
            print f
            if diff < thresh:
                break
            else:
                valid.pop(0)

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
        best = triangulate_add(zip(orignames,origcoordsflip), zip(matchnames,matchcoords), (nxtname,nxtposflip))[0]
        if not best:
            continue
        f,diff,diffs = best
        print f
        print 'error:', round(diff,6)
        
        #viewmatch(origcoordsflip, f)
        
        if diff < thresh and diff/prevdiff < 10: # stay within thresh and dont worsen more than 10x
            orignames.append(nxtname)
            origcoords.append(nxtpos)
            matchnames = f['properties']['combination']
            matchcoords = f['geometry']['coordinates'][0]
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)

def warp(image, tiepoints):
    import os
    print 'control points:', tiepoints
    gcptext = ' '.join('-gcp {0} {1} {2} {3}'.format(imgx,imgy,geox,geoy) for (imgx,imgy),(geox,geoy) in tiepoints)
    call = 'gdal_translate -of GTiff {gcptext} "{image}" "testmaps/warped.tif" & pause'.format(gcptext=gcptext, image=image)
    os.system(call)
    os.system('gdalwarp -r bilinear -tps -co COMPRESS=NONE -dstalpha -overwrite "testmaps/warped.tif" "testmaps/warped2.tif" & pause')

if __name__ == '__main__':

    # IDEA:
    # once first correct triangle matches found
    # add each subsequent point to that triangle and only keep if accuracy stays within threshold
    # no need to recalculate all possible combinations

    # manually set points

    #img = 'testmaps/indo_china_1886.jpg'
    #test = drawpoints(img)
    #test = [('Quedah', (781.3125996810211, 1495.2308612440197)), ('Malacca', (889.2041467304629, 1716.9142743221696)), ('Bankok', (785.5271132376399, 1038.3775917065395)), ('Saigon', (1140.3891547049443, 1201.057814992026)), ('Hanoi', (1071.2711323763958, 605.9685007974485)), ('Akyab', (353.1180223285493, 656.1212121212127)), ('Rangoon', (498.5187400318987, 863.264553429028)), ('Mandalay', (536.0279106858058, 548.6511164274324)), ('Yun-nan', (899.7404306220102, 365.9519537480064))]
    #test = [('Malacca', (889.2041467304629, 1716.9142743221696)), ('Bankok', (785.5271132376399, 1038.3775917065395)), ('Saigon', (1140.3891547049443, 1201.057814992026)), ('Hanoi', (1071.2711323763958, 605.9685007974485)), ('Rangoon', (498.5187400318987, 863.264553429028)), ('Mandalay', (536.0279106858058, 548.6511164274324)), ('Yun-nan', (899.7404306220102, 365.9519537480064))]
    #test.pop(5)
    #test.pop(0)
    #test.pop(-3)
    #test.pop(-1)
    
    #img = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #test = drawpoints(img)

    #img = 'testmaps/Kristiania_1887.jpg'
    #test = drawpoints(img)
    #test = [('Jernbanetorget, Oslo, Norway', (3057.0332062400316, 3332.1551908891547)), ('Kongelige slott, oslo, norway', (1939.121797747207, 3307.6237477571763)), ('Skarpsno, oslo, norway', (864.7800382525887, 4019.976221404005)), ('Karlsborg, oslo, norway', (3908.165170672597, 4175.037893194899)), ('Kampen, oslo, norway', (4308.306118078398, 2698.015504681892)), ('Sandaker, oslo, norway', (2661.237936260217, 445.9387253756717))]
    
    #img = 'testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg'
    #test = drawpoints(img)
    #test = [('Oro', (6273.619140625, 5655.9775390625)), ('Shaki', (4181.630078125, 5045.660058593749)), ('Yelwa', (6074.030078124998, 2027.337499999999))]

    #img = 'testmaps/2113087.jpg'
    #test = drawpoints(img)
    #test = [('Konakry', (546.2107421875002, 1232.299316406251)), ('Kidal', (1913.4460937499998, 414.70908203125106)), ('Sokoto', (2251.6786621093747, 912.4295166015633)), ('Ibadan', (2129.5353027343745, 1432.5704101562508)), ('Kandi', (2038.523828124999, 1104.3330566406257)), ('Timbuktu', (1528.21767578125, 563.3535156250005)), ('Farakan', (1168.6650390624998, 981.0893066406256)), ('Bingerville', (1409.8339843750002, 1626.5142578125015)), ('Port Harcourt', (2416.6914062499995, 1677.407324218752))]
    #test += [('Freetown', (587.10859375, 1330.05068359375)), ('Monrovia', (797.0081054687502, 1536.5802490234378)), ('Lagos', (2082.814550781249, 1520.487036132813)), ('Accra', (1744.0317871093746, 1603.4289794921879)), ("Fada N'Gourma", (1808.4046386718742, 999.9334960937498)), ('Bamako', (1073.8938964843753, 946.7020996093746)), ('Dakar', (254.58430175781353, 733.3638671874992))]

    img = 'testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg'
    #test = drawpoints(img)
    test = [('Accra', (4470.905029296875, 1914.4513549804688)), ('Sago', (469.5520833333328, 2105.3281249999995)), ('Agboville', (1657.0520833333333, 1637.5546875)), ('Grand Bassam', (2003.10107421875, 2160.2200927734375)), ('Grand Lahou', (1091.766845703125, 2198.0557861328125)), ('Beoumi', (698.2548828125, 413.63043212890625)), ('Zamaka', (2216.144287109375, 945.9882202148436)), ('Techiman', (3251.75830078125, 478.48791503906244)), ('Waso', (4551.898681640625, 534.1641845703125)), ('Obuasi', (3445.209228515625, 1443.5208740234375)), ('Nappa', (533.7742919921875, 1421.5924682617188)), ('Farakro', (1784.802734375, 326.31201171875))]
    #test.pop(-4)

    # process and warp
    origs,matches = process_optim(test, 0.4)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)
    warp(img, tiepoints)

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
    m.add_layer(anchors, fillcolor=(0,255,0), outlinewidth=0.1)
    
    m.zoom_bbox(*rlyr.bbox)
    m.zoom_out(2)
    m.view()

    fdsfsd


    # prespecified
##    test = [('Paris', (2.3514992, 48.8566101)),
##            ('Marseille', (5.3699525, 43.2961743)),
##             ('Oslo', (10.7389701, 59.9133301)),
##             ]
##    test = [('Berkeley', (-77.181000, 37.310193)),
##             ('Roxbury', (-77.142826, 37.458879)),
##              #('Wayside', (-77.194570, 37.377275)),
##              #('Garysville', (-77.154816, 37.246375)),
##              ('Huntington', (-82.457090, 38.403313)),
##             ]
##    test = [#('Carmel', (-86.058616, 39.976920)),
##            ('Bloomington', (-86.484200, 39.157321)), # indiana
##            ('Bloomington', (-88.935615, 40.446222)), # illinois
##             ('Springfield', (-89.585690, 39.780147)),
##             ]
##    test = [('Akwanga', (8.413046, 8.878104)),
##            ('Bida', (6.062102, 9.067167)),
##             ('Gali', (10.292387, 10.781456)),
##             ]
##    test = [('Telem', (35.031491, 31.561134)),
##             ('Nehusha', (34.954969, 31.624897)),
##              #('Lakhish', (34.849253, 31.557951)),
##              ('Hebron', (35.100556, 31.532643)),
##             ]
##    test = [('Tromso', (18.96240234375, 69.66472343054366)),
##            ('Trondheim', (9.931640625, 63.39152174400882)),
##             ('Oslo', (10.8544921875, 59.977005492196)),
##             ]
##    run(test)
    



        
