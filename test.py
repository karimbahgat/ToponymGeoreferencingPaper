"""
"""

from automap import triangulate, normalize

def drawpoints(imgpath):
    import pythongis as pg
    from pythongis.app import dialogs, icons

    import tk2

    points = []
    
    class ClickControl(tk2.basics.Label):
        def __init__(self, master, *args, **kwargs):
            tk2.basics.Label.__init__(self, master, *args, **kwargs)

            self.clickbut = tk2.basics.Button(self, command=self.begin_click)
            self.clickbut.set_icon(icons.iconpath("identify.png"), width=40, height=40)
            self.clickbut.pack()

            self.mouseicon_tk = icons.get("identify.png", width=30, height=30)

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
    feat['geometry']['coordinates'][0].append(feat['geometry']['coordinates'][0][0]) # close poly
    geoj = normalize(feat['geometry'])
    print 'red',geoj
    d.add_feature([], geoj)
    m.add_layer(d, fillcolor=None, outlinecolor='red')

    m.zoom_auto()
    m.zoom_out(1.2)
    m.view()

def run(test):
    names,positions = zip(*test)
    # reverse ys due to flipped image coordsys
    maxy = max((y for x,y in positions))
    positions = [(x,maxy-y) for x,y in positions]
    # triangulate
    matches = triangulate(names, positions)
    for f,diff,diffs in matches[:1]:
        print '-------'
        print 'error:', round(diff,6)
        for c in f['properties']['combination']:
            print c
        viewmatch(positions, f)
    return matches

def warp(image, tiepoints):
    import os
    print 'control points:', tiepoints
    gcptext = ' '.join('-gcp {0} {1} {2} {3}'.format(imgx,imgy,geox,geoy) for (imgx,imgy),(geox,geoy) in tiepoints)
    call = 'gdal_translate -of GTiff {gcptext} "{image}" "warped.tif" & pause'.format(gcptext=gcptext, image=image)
    os.system(call)
    os.system('gdalwarp -r bilinear -tps -co COMPRESS=NONE -dstalpha -overwrite "warped.tif" "warped2.tif" & pause')

if __name__ == '__main__':

    # manually set points

    img = 'testmaps/indo_china_1886.jpg'
    test = drawpoints(img)
    
    #img = 'testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
    #test = drawpoints(img)
    
    #img = 'testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg'
    #test = drawpoints(img)
    #test = [('Oro', (6273.619140625, 5655.9775390625)), ('Shaki', (4181.630078125, 5045.660058593749)), ('Yelwa', (6074.030078124998, 2027.337499999999))]
    
    matches = run(test)
    names,imgcoords = zip(*test)
    geocoords = matches[0][0]['geometry']['coordinates'][0]
    tiepoints = zip(imgcoords, geocoords)
    warp(img, tiepoints)

    # view warped
    import pythongis as pg
    m = pg.renderer.Map()
    m.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp")
    rlyr = m.add_layer('warped2.tif')
    m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_populated_places_simple\ne_10m_populated_places_simple.shp",
                fillcolor='red')
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
    



        
