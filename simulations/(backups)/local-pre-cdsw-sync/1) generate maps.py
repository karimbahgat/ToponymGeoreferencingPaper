
import pythongis as pg
import pycrs

import os
import sys
import datetime
from random import uniform, seed, randrange
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
# PARAMETERS
N = 48*4 # (48*4=192 scenes) (*96=18,432 maps) # number of unique map scenes to render (each is rendered almost a hundred times with different map parameters)
MAXPROCS = 4 # number of available cpu cores / parallel processes




####################
# FUNCTIONS

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
def _sampleplaces(projbox, projection, extent, n, distribution):
    '''
    Samples...
    '''
    if projection:
        lonlat = '+proj=longlat +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +nodef'
        bbox = project_bbox(projbox, projection, lonlat)
        backward = get_crs_transformer(projection, lonlat)
        forward = get_crs_transformer(lonlat, projection)
    else:
        bbox = projbox
        
    print('sampling places within',projbox,bbox)
    #hits = list(places.quick_overlap(bbox))
    hits = pg.VectorData(fields=['name'], crs=projection)
    db = sqlite3.connect('data/gns.db')
    x1,y1,x2,y2 = bbox
    projx1,projy1,projx2,projy2 = projbox
    projxmin,projymin,projxmax,projymax = min(projx1,projx2),min(projy1,projy2),max(projx1,projx2),max(projy1,projy2)
    for names,x,y in db.execute('select names,x,y from data where (x between ? and ?) and (y between ? and ?)', (x1,x2,y1,y2)):
        name = names.split('|')[1] if '|' in names else names
        try: name.encode('latin')
        except: continue
        
        # project placecoord and check within projected map bounds
        if projection:
            projx,projy = forward([(x,y)])[0]
            if not (projxmin <= projx <= projxmax) or not (projymin <= projy <= projymax):
                continue

        # add
        geoj = {'type':'Point', 'coordinates':(x,y)}
        hits.add_feature([name], geoj)
        
    print('possible places in map window',hits)

    if distribution == 'random':
        def samplefunc():
            while True:
                x = uniform(projbox[0], projbox[2])
                y = uniform(projbox[1], projbox[3])
                yield x,y

    elif distribution == 'dispersed':
        def samplefunc():
            while True:
                w = projbox[2]-projbox[0]
                h = projbox[3]-projbox[1]
                
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
                    y = projbox[1] + row*dy
                    y += dy/2.0
                    for col in range(columns):
                        x = projbox[0] + col*dx
                        x += dx/2.0
                        yield x,y

    else:
        raise Exception('Distribution type {} is not a valid option'.format(distribution))

    itersamples = samplefunc()
    results = []
    visited = set()
    hits = list(enumerate(hits)) 
    
    # loop all hits/samples
    while len(visited) < len(hits):
        x,y = next(itersamples)
        
        # reproj back to longlat
        if projection:
            x,y = backward([(x,y)])[0]
        
        #bufbox = [x-radius, y-radius, x+radius, y+radius]
        def dist(h):
            i,f = h
            fx,fy = f.geometry['coordinates']
            return math.hypot(x-fx, y-fy)
        #print 'dist sorting'
        sortplaces = sorted(hits, key=dist)
        #print 'done'
        for i,f in sortplaces: 
            #print '---'
            #print('attempt to sample place near', (x,y), f.row, 'of', len(hits) )
            visited.add(i)
            # compare with previous results
            sameflag = False
            for f2 in results:
                if f == f2:
                    # dont yield same place multiple times
                    sameflag = True
                    break
                fx,fy = f.geometry['coordinates']
                fx2,fy2 = f2.geometry['coordinates']
                nearthresh = (extent / 100.0) * 5 #(extent * 100 * 1000) / 100.0 # extent dec deg * 100 = ca km * 1000 = ca m / 100 = 1% of extent in projected meters
                #print('nearness:', math.hypot(fx-fx2, fy-fy2), 'vs', nearthresh)
                if math.hypot(fx-fx2, fy-fy2) < nearthresh:
                    # dont yield places that are approximately the same (<5% of map extent)
                    sameflag = True
                    break
            if sameflag:
                #print('ignore sample, same or too close')
                continue
            # all tests passed, yield this sample
            results.append(f)
            print('sampling', len(results), 'of', n)
            yield f
            # yield only closest valid match, then break
            break
        # stop if reached quota
        if len(results) >= n:
            break

def get_crs_transformer(fromcrs, tocrs):
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
        def _isvalid(p):
            x,y = p
            return not (math.isinf(x) or math.isnan(x) or math.isinf(y) or math.isnan(y))
        def _project(points):
            xs,ys = itertools.izip(*points)
            xs,ys = pyproj.transform(fromcrs,
                                     tocrs,
                                     xs, ys)
            newpoints = list(itertools.izip(xs, ys))
            newpoints = [p for p in newpoints if _isvalid(p)] # drops inf and nan
            return newpoints
    else:
        _project = None

    return _project

def project_bbox(bbox, fromcrs, tocrs, sampling_freq=10):
    print 'projecting bbox', bbox, fromcrs, tocrs
    transformer = get_crs_transformer(fromcrs, tocrs)
    if not transformer:
        return bbox
    x1,y1,x2,y2 = bbox
    w,h = x2-x1, y2-y1
    sampling_freq = int(sampling_freq)
    dx,dy = w/float(sampling_freq), h/float(sampling_freq)
    gridsamples = [(x1+dx*ix,y1+dy*iy)
                   for iy in range(sampling_freq+1)
                   for ix in range(sampling_freq+1)]
    gridsamples = transformer(gridsamples)
    if not gridsamples:
        print 'no valid gridsamples'
        return None
    xs,ys = zip(*gridsamples)
    xmin,ymin,xmax,ymax = min(xs),min(ys),max(xs),max(ys)
    bbox = [xmin,ymin,xmax,ymax] 
    print '-->',bbox
    return bbox


# get map places
def get_mapplaces(projbox, projection, extent, quantity, distribution, uncertainty):
    '''
    - quantity: aka number of placenames
    - distribution: how placenames are distributed across map
        - random
        - dispersed
        - clustered
    - uncertainty: random placename coordinate displacement (in decimal degrees)
    '''
    # get places to be rendered in map
    mapplaces = pg.VectorData()
    mapplaces.fields = ['name']
    for f in _sampleplaces(projbox, projection, extent, quantity, distribution):
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

    # country background
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

    # extra data
    for datadef in datas:
        if datadef:
            data,style = datadef
            m.add_layer(data, **style)

    # toponyms
    m.add_layer(mapplaces,
                text=lambda f: f['name'],
                textoptions=textopts,
                legendoptions={'title':'Populated place'},
                **anchoropts)

    # zoom
    m.zoom_bbox(*bbox, geographic=True)

    # gridlines
    if datas:
        gridlines = pg.VectorData()
        if projection:
            projbox = m.bbox
            projbox = [ projbox[0],projbox[3],projbox[2],projbox[1] ]
            lonlat = '+proj=longlat +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +nodef'
            _bbox = project_bbox(projbox, projection, lonlat)
        else:
            _bbox = bbox
        bw,bh = _bbox[2]-_bbox[0], _bbox[3]-_bbox[1]
        dx,dy = bw/4.0, bh/4.0
        for gxi in range(4+1):
            gx = _bbox[0]+dx*gxi
            vert = [(gx, _bbox[1]+bh/20.0*gyi) for gyi in range(20+1)]
            geoj = {'type':'LineString', 'coordinates':vert}
            gridlines.add_feature([], geoj)
        for gyi in range(4+1):
            gy = _bbox[1]+dy*gyi
            horiz = [(_bbox[0]+bw/20.0*gxi, gy) for gxi in range(20+1)]
            geoj = {'type':'LineString', 'coordinates':horiz}
            gridlines.add_feature([], geoj)
        print 'gridlines',gridlines
        m.add_layer(gridlines, fillcolor=(62,88,130), fillsize=0.15)

    # legend
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

def getscene(extent, quantity, projection):
    # find a center coordinate suitable for these scene params
    attempts = 0
    while attempts < 20:
        attempts += 1

        if extent > 1:
            center = (uniform(-170,170),uniform(-70,70))
        else:
            idx = randrange(1, len(bigcities)+1)
            city = bigcities[idx]
            center = city.geometry['coordinates']
        regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
        bbox = mapregion(**regionopts)
        print('attempt', attempts, ': trying scene centered at', center)

        # customize projection to area
        if not projection:
            proj = projection
        elif 'lcc' in projection:
            bh = bbox[3]-bbox[1]
            projparams = dict(lon_0=center[0],
                              lat_0=center[1],
                            lat_1=bbox[1]+bh*0.25,
                            lat_2=bbox[1]+bh*0.75)
            proj = projection + ' +lon_0={lon_0} +lat_0={lat_0} +lat_1={lat_1} +lat_2={lat_2}'.format(**projparams)
        elif 'tmerc' in projection:
            projparams = dict(lon_0=center[0],
                              lat_0=center[1])
            proj = projection + ' +lon_0={lon_0} +lat_0={lat_0}'.format(**projparams)
        elif 'eqc' in projection:
            projparams = dict(lon_0=center[0],
                              lat_0=center[1],
                              lat_ts=center[1])
            proj = projection + ' +lon_0={lon_0} +lat_0={lat_0} +lat_ts={lat_ts}'.format(**projparams)
        print proj

        # get map projected bbox
        if proj:
            m = pg.renderer.Map(100, int(100*regionopts['aspect']), crs=proj)
            m.zoom_bbox(*bbox, geographic=True)
            projbox = m.bbox
            projbox = [ projbox[0],projbox[3],projbox[2],projbox[1] ]
        else:
            projbox = bbox

        # check enough land before sending to process
        bbox = mapregion(**regionopts)
        if not valid_mapregion(bbox):
            print('!!! Not enough land area, skipping')
            continue

        # get placenames within map bbox
        placeopts = {'quantity':quantity, 'distribution':'dispersed', 'uncertainty':0} # hardcoded just to test available places...
        mapplaces = get_mapplaces(projbox, proj, extent, **placeopts)

        # check enough placenames
        if len(mapplaces) < quantity:
            print('!!! Not enough places, skipping')
            continue

        # check sufficient distribution
        ul,ur,ll,lr = 0,0,0,0 # quadrant counters
        cx,cy = regionopts['center']
        for f in mapplaces:
            x,y = f.geometry['coordinates']
            # right
            if x >= cx:
                # upper
                if y >= cy:
                    ur += 1
                # lower
                else:
                    lr += 1
            # left
            else:
                # upper
                if y >= cy:
                    ul += 1
                # lower
                else:
                    ll += 1
        thresh_quadrants = [True for q in (ul,ur,ll,lr) if q > 2]
        if len(thresh_quadrants) < 3:
            print('!!! Too poor toponym distribution, skipping')
            continue

        return center,extent,quantity,proj

        break # attempt successfull, exit while loop

def iterscenes():
    # loop scene params
    print('scenes at a time', len(list(itertools.product(extents, quantities, projections))))
    for extent,quantity,projection in itertools.product(extents, quantities, projections):

        # find a center coordinate suitable for these scene params
        attempts = 0
        while attempts < 20:
            attempts += 1

            if extent > 1:
                center = (uniform(-170,170),uniform(-70,70))
            else:
                idx = randrange(1, len(bigcities)+1)
                city = bigcities[idx]
                center = city.geometry['coordinates']
            regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
            bbox = mapregion(**regionopts)
            print('attempt', attempts, ': trying scene centered at', center)

            # customize projection to area
            if not projection:
                proj = projection
            elif 'lcc' in projection:
                bh = bbox[3]-bbox[1]
                projparams = dict(lon_0=center[0],
                                  lat_0=center[1],
                                lat_1=bbox[1]+bh*0.25,
                                lat_2=bbox[1]+bh*0.75)
                proj = projection + ' +lon_0={lon_0} +lat_0={lat_0} +lat_1={lat_1} +lat_2={lat_2}'.format(**projparams)
            elif 'tmerc' in projection:
                projparams = dict(lon_0=center[0],
                                  lat_0=center[1])
                proj = projection + ' +lon_0={lon_0} +lat_0={lat_0}'.format(**projparams)
            elif 'eqc' in projection:
                projparams = dict(lon_0=center[0],
                                  lat_0=center[1],
                                  lat_ts=center[1])
                proj = projection + ' +lon_0={lon_0} +lat_0={lat_0} +lat_ts={lat_ts}'.format(**projparams)
            print proj

            # get map projected bbox
            if proj:
                m = pg.renderer.Map(100, int(100*regionopts['aspect']), crs=proj)
                m.zoom_bbox(*bbox, geographic=True)
                projbox = m.bbox
                projbox = [ projbox[0],projbox[3],projbox[2],projbox[1] ]
            else:
                projbox = bbox

            # check enough land before sending to process
            bbox = mapregion(**regionopts)
            if not valid_mapregion(bbox):
                print('!!! Not enough land area, skipping')
                continue

            # get placenames within map bbox
            placeopts = {'quantity':quantity, 'distribution':'dispersed', 'uncertainty':0} # hardcoded just to test available places...
            mapplaces = get_mapplaces(projbox, proj, extent, **placeopts)

            # check enough placenames
            if len(mapplaces) < quantity:
                print('!!! Not enough places, skipping')
                continue

            # check sufficient distribution
            ul,ur,ll,lr = 0,0,0,0 # quadrant counters
            cx,cy = regionopts['center']
            for f in mapplaces:
                x,y = f.geometry['coordinates']
                # right
                if x >= cx:
                    # upper
                    if y >= cy:
                        ur += 1
                    # lower
                    else:
                        lr += 1
                # left
                else:
                    # upper
                    if y >= cy:
                        ul += 1
                    # lower
                    else:
                        ll += 1
            thresh_quadrants = [True for q in (ul,ur,ll,lr) if q > 2]
            if len(thresh_quadrants) < 3:
                print('!!! Too poor toponym distribution, skipping')
                continue

            yield center,extent,quantity,proj

            break # attempt successfull, exit while loop


def iteroptions(center, extent, quantity, projection):
    regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}

    # get map projected bbox
    bbox = mapregion(**regionopts)
    if projection:
        m = pg.renderer.Map(100, int(100*regionopts['aspect']), crs=projection)
        m.zoom_bbox(*bbox, geographic=True)
        projbox = m.bbox
        projbox = [ projbox[0],projbox[3],projbox[2],projbox[1] ]
    else:
        projbox = bbox
            
    # loop placename options
    for distribution,uncertainty in itertools.product(distributions,uncertainties):

        # get placenames within map bbox
        placeopts = {'quantity':quantity, 'distribution':distribution, 'uncertainty':uncertainty}
        mapplaces = get_mapplaces(projbox, projection, extent, **placeopts)

        # loop rendering options
        for datas,meta in itertools.product(alldatas,metas):

            metaopts = {'title':meta['title'], 'titleoptions':meta.get('titleoptions', {}), 'legend':meta['legend'], 'legendoptions':meta.get('legendoptions', {}), 'arealabels':meta['arealabels']}
            textopts = {'textsize':8, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
            anchoropts = {'fillcolor':'black', 'fillsize':0.1}
            resolution = resolutions[0] # render at full resolution (downsample later)

            yield regionopts,bbox,placeopts,mapplaces,datas,projection,metaopts,textopts,anchoropts,resolution

def run(i, extent, quantity, projection):
    # each run is on a scene
    # a scene is a combination of extent,quantity,projection, rendered at a center that can fullfill the quantity requirement

    seed(i) # random seed for replication purposes
    center,extent,quantity,projection = getscene(extent,quantity,projection)

    subi = 1
    for opts in iteroptions(center, extent, quantity, projection):
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

            #break # JUST FOR TESTING, REMOVE!!

        subi += 1

        #break # JUST FOR TESTING, REMOVE!!

def process(i, extent, quantity, projection):
    logger = codecs.open('maps/sim_{}_log.txt'.format(i), 'w', encoding='utf8', buffering=0)
    sys.stdout = logger
    sys.stderr = logger
    print('PID:',os.getpid())
    print('time',datetime.datetime.now().isoformat())
    print('working path',os.path.abspath(''))

    run(i, extent, quantity, projection)

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
bigcities = pg.VectorData("data/ne_10m_populated_places.shp")
bigcities.rename_field('NAME', 'name')
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
n = N # each N is a particular scene at a particular extent, toponym quantity, and projection
extents = [10] + [50, 1, 0.25] # ca 5000km, 1000km, 100km, and 25km
quantities = [80, 40, 20, 10]
distributions = ['random'] # MOST MAPS WILL RESEMBLE RANDOM  #['dispersed','random'] # IMPROVE W NUMERIC
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
alldatas = [
                [], #(roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),], # no data layers
                [
                (rivers, {'fillcolor':(54,115,159), 'fillsize':0.08, 'legendoptions':{'title':'Rivers'}}), # three layers
                (urban, {'fillcolor':(209,194,151), 'outlinecolor':(209-50,194-50,151-50), 'legendoptions':{'title':'Urban area'}}),
                (roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),
                # + gridlines (hardcoded in function above)
                 ],
            ]
projections = [#None, # lat/lon
               #'+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', #'+init=EPSG:3857', # Web Mercator
               #'+proj=moll +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54009', # World Mollweide
               #'+proj=robin +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54030', # Robinson
               '+proj=eqc +datum=WGS84 +ellps=WGS84 +units=m',
               '+proj=lcc +datum=WGS84 +ellps=WGS84 +units=m',
               '+proj=tmerc +datum=WGS84 +ellps=WGS84 +units=m',
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

    maxprocs = MAXPROCS
    procs = []

    print('combinations per scene', len(list(itertools.product(distributions,uncertainties,alldatas,metas,resolutions,imformats))))

    # begin
    
    # loop scene params
    i = 1
    while i < n:
        print('scenes at a time', len(list(itertools.product(extents, quantities, projections))))
        for extent,quantity,projection in itertools.product(extents, quantities, projections):
            #for i,(center,extent,quantity,projection) in enumerate(iterscenes()):
            
            print('-------')
            print('SCENE:',i,extent,quantity,projection)

            # test view
    ##            m = pg.renderer.Map(crs=projections[0])
    ##            for d in alldatas[1]:
    ##                m.add_layer(d[0], **d[1])
    ##            m.zoom_bbox(*bbox, geographic=True)
    ##            m.view()
    ##            i+=1
    ##            continue
            
            #run(i,center,extent)

            # Begin process
            p = mp.Process(target=process,
                           args=(i, extent, quantity, projection),
                           )
            p.start()
            procs.append(p)

            i += 1

            # Wait for next available process
            while len(procs) >= maxprocs:
                for p in procs:
                    if not p.is_alive():
                        procs.remove(p)

    # Wait for last process
    for p in procs:
        p.join()

                        

        




        



