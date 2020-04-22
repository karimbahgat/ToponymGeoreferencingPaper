
import pythongis as pg

import os
import sys
import datetime
from random import uniform, seed
import math
import json
import codecs
import itertools
import multiprocessing as mp
import sqlite3


print(os.getcwd())
try:
    os.chdir('simulations')
except:
    pass


####################
# FUNCTIONS

seed(16) # random seed for replication purposes

# select map region
def mapregion(center, extent, aspect):
    '''
    - center: lon,lat map center
    - extent: width of map in decimal degrees
    - aspect: height to width ratio
    '''
    lon,lat = center
    width = extent
    height = width * aspect
    bbox = [lon-width/2.0, lat-height/2.0, lon+width/2.0, lat+height/2.0]
    return bbox

# check valid map region
def valid_mapregion(bbox):
    bw = abs(bbox[0]-bbox[2])
    bh = abs(bbox[1]-bbox[3])
    window_area = bw * bh
    crop = countries.manage.crop(bbox)
    crop = crop.aggregate(lambda f: 1, 'union')
    if len(crop):
        land = list(crop)[0].get_shapely()
        if (land.area / window_area) > 0.5:
            return True

# sample n places within focus region
def _sampleplaces(bbox, n, distribution):
    '''
    Samples...
    '''
    print('sampling places within',bbox)
    #hits = list(places.quick_overlap(bbox))
    hits = pg.VectorData(fields=['name'])
    db = sqlite3.connect('data/gns.db')
    x1,y1,x2,y2 = bbox
    for names,x,y in db.execute('select names,x,y from data where (x between ? and ?) and (y between ? and ?)', (x1,x2,y1,y2)):
        name = names.split('|')[1] if '|' in names else names
        try: name.encode('latin')
        except: continue
        geoj = {'type':'Point', 'coordinates':(x,y)}
        hits.add_feature([name], geoj)
    print('possible places in map window',hits)

    if distribution == 'random':
        def samplefunc():
            while True:
                x = uniform(bbox[0], bbox[2])
                y = uniform(bbox[1], bbox[3])
                yield x,y

    elif distribution == 'dispersed':
        def samplefunc():
            while True:
                w = bbox[2]-bbox[0]
                h = bbox[3]-bbox[1]
                
        ##        aspect_ratio = h/float(w)
        ##        columns = math.sqrt(n/float(aspect_ratio))
        ##        if not columns.is_integer():
        ##            columns += 1
        ##        columns = int(round(columns))
        ##        rows = n/float(columns)
        ##        if not rows.is_integer():
        ##            rows += 1
        ##        rows = int(round(rows))

                rows = columns = int(round(math.sqrt(n)))
                
                #print rows,columns

                dx = w / float(columns)
                dy = h / float(rows)
                    
                for row in range(rows):
                    y = bbox[1] + row*dy
                    y += dy/2.0
                    for col in range(columns):
                        x = bbox[0] + col*dx
                        x += dx/2.0
                        yield x,y

    else:
        raise Exception('Distribution type {} is not a valid option'.format(distribution))

    itersamples = samplefunc()
    results = []

    if len(hits) > n:
        #radius = 2.0
        i = 0
        while True:
            x,y = next(itersamples) 
            #bufbox = [x-radius, y-radius, x+radius, y+radius]
            def dist(f):
                fx,fy = f.geometry['coordinates']
                return math.hypot(x-fx, y-fy)
            sortplaces = sorted(hits, key=dist)
            for f in sortplaces: #intersection('geom', bufbox):
                #print '---'
                #print x,y
                #print bufbox
                #print r.bbox
                print('checking place', (x,y), f.row )
                if f in results:
                    # dont yield same place multiple times
                    continue
                i += 1
                results.append(f)
                yield f
                # yield only first match inside bbox, then break
                break
            if i >= n:
                break

    else:
        for f in hits:
            yield f

def project_bbox(bbox, fromcrs, tocrs):
    '''Take a bbox from one crs, convert to bbox in another crs, then convert back to the original crs'''
    def get_crs_transformer(fromcrs, tocrs):
        import pycrs
        
        if not (fromcrs and tocrs):
            return None
        
        if isinstance(fromcrs, basestring):
            fromcrs = pycrs.parse.from_unknown_text(fromcrs)

        if isinstance(tocrs, basestring):
            tocrs = pycrs.parse.from_unknown_text(tocrs)

        fromcrs = fromcrs.to_proj4()
        tocrs = tocrs.to_proj4()
        
        if fromcrs != tocrs:
            import pyproj
            fromcrs = pyproj.Proj(fromcrs)
            tocrs = pyproj.Proj(tocrs)
            def _project(points):
                xs,ys = itertools.izip(*points)
                xs,ys = pyproj.transform(fromcrs,
                                         tocrs,
                                         xs, ys)
                newpoints = list(itertools.izip(xs, ys))
                return newpoints
        else:
            _project = None

        return _project

    _transform = get_crs_transformer(fromcrs, tocrs)
    x1,y1,x2,y2 = bbox
    corners = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
    corners = _transform(corners)
    xs,ys = zip(*corners)
    xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs),max(ys)
    projbox = [xmin,ymin,xmax,ymax] 
    return projbox


# get map places
def get_mapplaces(bbox, quantity, distribution, uncertainty):
    '''
    - quantity: aka number of placenames
    - distribution: how placenames are distributed across map
        - random
        - dispersed
        - clustered
    - hierarchy ??? :
        - only big places
        - any small or big place
    '''
    # get places to be rendered in map
    mapplaces = pg.VectorData()
    mapplaces.fields = ['name']
    for f in _sampleplaces(bbox, quantity, distribution):
        #print f
        name = f['name'].title() #r['names'].split('|')[0]
        row = [name]
        x,y = f.geometry['coordinates']
        if uncertainty:
            x += uniform(-uncertainty, uncertainty)
            y += uniform(-uncertainty, uncertainty)
        geoj = {'type':'Point', 'coordinates':(x,y)}
        mapplaces.add_feature(row, geoj) #r['geom'].__geo_interface__)

    if len(mapplaces):
        mapplaces.create_spatial_index()
            
    return mapplaces

# render map
def render_map(bbox, mapplaces, datas, resolution, regionopts, projection, anchoropts, textopts, metaopts):
    # determine resolution
    width = resolution
    height = int(width * regionopts['aspect'])
    
    # render pure map image
    m = pg.renderer.Map(width, height,
                        title=metaopts['title'],
                        titleoptions=metaopts['titleoptions'],
                        #textoptions={'font':'eb garamond 12'},
                        #textoptions={'font':'lato'},
                        #textoptions={'font':'dejavu sans'},
                        #textoptions={'font':'freesans'},
                        background=(91,181,200),
                        crs=projection)
    if metaopts['arealabels']:
        arealabels = {'text':lambda f: f['NAME'].upper(), 'textoptions': {'textsize':textopts['textsize']*1.5, 'textcolor':(88,88,88)}}
        rencountries = countries #.manage.crop(bbox)
        #rencountries.create_spatial_index()
    else:
        arealabels = {}
        rencountries = countries
    m.add_layer(rencountries, fillcolor=(255,222,173), outlinewidth=0.2, outlinecolor=(100,100,100),
                legendoptions={'title':'Country border'},
                **arealabels)

    for datadef in datas:
        if datadef:
            data,style = datadef
            m.add_layer(data, **style)
        
    m.add_layer(mapplaces,
                text=lambda f: f['name'],
                textoptions=textopts,
                legendoptions={'title':'Populated place'},
                **anchoropts)
    m.zoom_bbox(*bbox, geographic=True)

    if metaopts['legend']:
        legendoptions = {'padding':0, 'direction':'s'}
        legendoptions.update(metaopts['legendoptions'])
        m.add_legend(legendoptions=legendoptions)

    # note...

    m.render_all(antialias=True)

    return m

# save map
def save_map(name, mapp, mapplaces, datas, resolution, regionopts, placeopts, projection, anchoropts, textopts, metaopts, noiseopts):
    print('SAVING:',name)
    
    # downscale to resolution
    width,height = mapp.width, mapp.height
    ratio = noiseopts['resolution'] / float(width)
    newwidth = noiseopts['resolution']
    newheight = int(height * ratio)

    # save
    m = mapp
    imformat = noiseopts['format']
    saveargs = {}
    if imformat == 'jpg':
        saveargs['quality'] = 10

    # downsample to resolution
    #img = m.img.resize((newwidth, newheight), 1) # resize img separately using nearest neighbour resampling, since map resize uses antialias
    m = m.copy()
    m.resize(newwidth, newheight)

    # store rendering with original geo coordinates
    m.save('maps/{}_image.{}'.format(name, imformat), meta=True, **saveargs)
    #r = pg.RasterData(image=img, crs) 
    #r.set_geotransform(affine=m.drawer.coordspace_invtransform) # inverse bc the drawer actually goes from coords -> pixels, we need pixels -> coords
    #r.save('maps/{}_image.{}'.format(name, imformat), **saveargs)

    # store the original place coordinates
    mapplaces = mapplaces.copy()
    #if projection:
    #    mapplaces.manage.reproject(projection)
    mapplaces.add_field('col')
    mapplaces.add_field('row')
    mapplaces.add_field('x')
    mapplaces.add_field('y')
    for f in mapplaces:
        x,y = f.geometry['coordinates']
        col,row = m.drawer.coord2pixel(x,y)
        f['col'] = col
        f['row'] = row
        f['x'] = x
        f['y'] = y
    mapplaces.save('maps/{}_placenames.geojson'.format(name))

    # save options as json
    opts = dict(name=name,
              resolution=resolution,
              regionopts=regionopts,
              placeopts=placeopts,
              projection=projection,
              anchoropts=anchoropts,
              textopts=textopts,
              metaopts=metaopts,
              noiseopts=noiseopts,
              datas=[datadef[-1] if datadef else None
                     for datadef in datas],
              )
    with open('maps/{}_opts.json'.format(name), 'w') as fobj:
        fobj.write(json.dumps(opts))

def iteroptions(center, extent):
    regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
    bbox = mapregion(**regionopts)

    # loop placename options
    for quantity,distribution,uncertainty in itertools.product(quantities,distributions,uncertainties):

        # FIX PROJECTION...
        for projection in projections:

##            if projection:
##                lonlat = '+proj=longlat +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +nodef'
##                projbox = project_bbox(bbox, lonlat, projection)
##                placebox = project_bbox(projbox, projection, lonlat)
##                print('bbox to projbox and back',bbox,placebox)
##            else:
##                placebox = bbox

            # check enough placenames
            placeopts = {'quantity':quantity, 'distribution':distribution, 'uncertainty':uncertainty}
            mapplaces = get_mapplaces(bbox, **placeopts)
            if len(mapplaces) < 10: #quantity:
                print('!!! Not enough places, skipping')
                continue

            # loop rendering options
            for datas,meta in itertools.product(alldatas,metas):

                #projbox = project_bbox(bbox)
                
                metaopts = {'title':meta['title'], 'titleoptions':meta.get('titleoptions', {}), 'legend':meta['legend'], 'legendoptions':meta.get('legendoptions', {}), 'arealabels':meta['arealabels']}
                textopts = {'textsize':8, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
                anchoropts = {'fillcolor':'black', 'fillsize':0.1}
                resolution = resolutions[0] # render at full resolution (downsample later)

                yield regionopts,bbox,placeopts,mapplaces,datas,projection,metaopts,textopts,anchoropts,resolution

def run(i, center, extent):
    subi = 1
    for opts in iteroptions(center, extent):
        regionopts,bbox,placeopts,mapplaces,datas,projection,metaopts,textopts,anchoropts,resolution = opts

        # render the map
        mapp = render_map(bbox,
                         mapplaces,
                         datas,
                         resolution,
                         regionopts=regionopts,
                         projection=projection,
                         anchoropts=anchoropts,
                         textopts=textopts,
                         metaopts=metaopts,
                         )

        # loop image save options
        subsubi = 1
        for resolution,imformat in itertools.product(resolutions,imformats):

            name = 'sim_{}_{}_{}'.format(i, subi, subsubi)
            
            noiseopts = {'resolution':resolution, 'format':imformat}
            
            save_map(name, mapp, mapplaces, datas, resolution, regionopts, placeopts, projection, anchoropts, textopts, metaopts, noiseopts)
            
            subsubi += 1

        subi += 1

def process(i, center, extent):
    logger = codecs.open('maps/sim_{}_log.txt'.format(i), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))

    run(i, center, extent)

    print('process finished, should exit')
        

#######################
# MISC TESTING

# Test sampling distribution
##def samplefunc(bbox, n):
##    if True:
##        w = bbox[2]-bbox[0]
##        h = bbox[3]-bbox[1]
##        
####        aspect_ratio = h/float(w)
####        columns = math.sqrt(n/float(aspect_ratio))
####        if not columns.is_integer():
####            columns += 1
####        columns = int(round(columns))
####        rows = n/float(columns)
####        if not rows.is_integer():
####            rows += 1
####        rows = int(round(rows))
##
##        rows = columns = int(round(math.sqrt(n)))
##        
##        print rows,columns
##
##        dx = w / float(columns)
##        dy = h / float(rows)
##
####        r = pg.RasterData(mode='1bit', width=columns, height=rows,
####                          xscale=dx, yscale=dy, xoffset=bbox[0], yoffset=bbox[2])
####        r.add_band()
####        for cell in r.bands[0]:
####            yield cell.point['coordinates']
##            
##        for row in range(rows):
##            y = bbox[1] + row*dy
##            y += dy/2.0
##            for col in range(columns):
##                x = bbox[0] + col*dx
##                x += dx/2.0
##                yield x,y
##
##import pyagg
##c=pyagg.Canvas(1000,500)
##bbox=0,0,100,100
##c.custom_space(*bbox)
##for x,y in samplefunc(bbox, 40):
##    print x,y
##    c.draw_circle((x,y))
##c.view()
##
##sdfsdf




####################
# RUN
pg.vector.data.DEFAULT_SPATIAL_INDEX = 'quadtree'

# load data (all processes)
print('loading data')
countries = pg.VectorData("data/ne_10m_admin_0_countries.shp")
countries.create_spatial_index()
#places = pg.VectorData("data/ne_10m_populated_places.shp")
#places.rename_field('NAME', 'name')
#places = pg.VectorData("data/global_settlement_points_v1.01.shp", encoding='latin')
#places.rename_field('Name1', 'name')
#places.create_spatial_index()
rivers = pg.VectorData("data/ne_10m_rivers_lake_centerlines.shp") 
rivers.create_spatial_index()
urban = pg.VectorData("data/ne_10m_urban_areas.shp") 
urban.create_spatial_index()
roads = pg.VectorData("data/ne_10m_roads.shp") 
roads.create_spatial_index()

# options, NEW

### text error
##noise = imformat
##mapnoise = datas
##
### error magnitude
##scale = extent
##resolution
##
### warp error
##information = quantity
##distribution
##uncertainty = placeerror
##textnoise = meta
##distortion = projection

# options
print('defining options')
n = 25 # with 4 extents for each = 40
extents = [10] + [50, 1, 0.1] # ca 5000km, 1000km, 100km, and 10km
quantities = [80, 40, 20, 10]
distributions = ['dispersed', 'random'] # IMPROVE W NUMERIC
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
alldatas = [
                [], #(roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),], # no data layers
                [
                (rivers, {'fillcolor':(54,115,159), 'fillsize':0.08, 'legendoptions':{'title':'Rivers'}}), # three layers
                (urban, {'fillcolor':(209,194,151), 'legendoptions':{'title':'Urban area'}}),
                (roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),
                 ],
            ]
projections = [None, # lat/lon
               #'+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', #'+init=EPSG:3857', # Web Mercator
               #'+proj=moll +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54009', # World Mollweide
               '+proj=robin +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54030', # Robinson
               ]
resolutions = [3000, 2000, 1000] #, 750] #, 4000]
imformats = ['png','jpg']
metas = [{'title':'','legend':False,'arealabels':False}, # nothing
         {'title':'This is the Map Title','titleoptions':{'fillcolor':'white'},'legend':True,'legendoptions':{'fillcolor':'white'},'arealabels':True}, # text noise + meta boxes (arealabels + title + legend)
         #{'title':'This is the Map Title','titleoptions':{'fillcolor':None},'legend':True,'legendoptions':{'fillcolor':None},'arealabels':True}, # text noise (arealabels + title + legend)
         #{'title':'This is the Map Title','titleoptions':{'fillcolor':'white'},'legend':True,'legendoptions':{'fillcolor':'white'},'arealabels':False}, # meta boxes (title + legend)
         ]

# main process handler
if __name__ == '__main__':

    maxprocs = 2
    procs = []

    print('combinations per region', len(list(itertools.product(quantities,distributions,uncertainties,alldatas,projections,metas,resolutions,imformats))))

    # begin
    
    # zoom regions
    i = 1
    while i < n:
        center = (uniform(-170,170),uniform(-70,70))
        for extent in extents:
            print('-------')
            print('REGION:',i,center,extent)

            # check enough land before sending to process
            regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
            bbox = mapregion(**regionopts)
            if not valid_mapregion(bbox):
                print('!!! Not enough land area, skipping')
                continue
            
            #run(i,center,extent)

            # Begin process
            p = mp.Process(target=process,
                           args=(i, center, extent),
                           )
            p.start()
            procs.append(p)

            # Wait for next available process
            while len(procs) >= maxprocs:
                for p in procs:
                    if not p.is_alive():
                        procs.remove(p)

            i += 1
                        

        




        



