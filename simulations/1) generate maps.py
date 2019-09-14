
import pythongis as pg

from random import uniform
import math
import json



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
def _sampleplaces(bbox, n, distribution):
    '''
    Samples...
    '''
    print('target box',bbox)
    hits = list(places.quick_overlap(bbox))

    if len(hits) > n:
        radius = 0.5
        i = 0
        while True:
            if distribution == 'random':
                x = uniform(bbox[0], bbox[2])
                y = uniform(bbox[1], bbox[3])
            else:
                raise Exception('Distribution type {} is not a valid option'.format(distribution))
            bufbox = [x-radius, y-radius, x+radius, y+radius]
            def dist(f):
                fx,fy = f.geometry['coordinates']
                return math.hypot(x-fx, y-fy)
            sortplaces = sorted(places.quick_overlap(bufbox), key=dist)
            for r in sortplaces: #intersection('geom', bufbox):
                #print '---'
                #print x,y
                #print bufbox
                #print r.bbox
                i += 1
                yield r
                # yield only first match inside bbox, then break
                break
            if i >= n:
                break

    else:
        for r in hits:
            yield r

# get map places
def get_mapplaces(bbox, quantity, distribution):
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
        name = f['Name1'].title() #r['names'].split('|')[0]
        row = [name]
        mapplaces.add_feature(row, f.geometry) #r['geom'].__geo_interface__)
            
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
                        background=(91,181,200),
                        crs=projection)
    if metaopts['arealabels']:
        arealabels = {'text':lambda f: f['NAME'].upper(), 'textoptions': {'textsize':textopts['textsize']*1.5, 'textcolor':(88,88,88)}}
        rencountries = countries.manage.crop(bbox)
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
    m.zoom_bbox(*bbox)

    if metaopts['legend']:
        m.add_legend(legendoptions={'padding':0, 'direction':'s'})

    # note...

    m.render_all(antialias=True)

    return m

# save map
def save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts):
    print('SAVING:',name)
    
    # downscale to resolution
    width,height = mapp.width, mapp.height
    ratio = noiseopts['resolution'] / float(width)
    newwidth = noiseopts['resolution']
    newheight = int(height * ratio)

    # save
    m = mapp
    imformat = noiseopts['format']
    m.img.convert('RGB').save('maps/{}_image.{}'.format(name, imformat))

    m.img.resize((newwidth, newheight), 1).convert('RGB').save('maps/{}_image.{}'.format(name, imformat))

    # store rendering with original geo coordinates
    affine = m.drawer.coordspace_transform
    r = pg.RasterData('maps/{}_image.{}'.format(name, imformat)) 
    r.set_geotransform(affine=m.drawer.coordspace_invtransform)
    r.save('maps/{}_truth.tif'.format(name))

    # store the original place coordinates
    # WARNING: NOT CORRECT FOR THE LOWER RESOLUTIONS...
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
              projection=projection,
              anchoropts=anchoropts,
              textopts=textopts,
              metaopts=metaopts,
              noiseopts=noiseopts,
              )
    with open('maps/{}_opts.json'.format(name), 'w') as fobj:
        fobj.write(json.dumps(opts))






####################
# RUN

fdasfasfafas

if __name__ == '__main__':
    import itertools
    
    # load data
    countries = pg.VectorData("data/ne_10m_admin_0_countries.shp")
    #places = pg.VectorData("data/ne_10m_populated_places.shp")
    places = pg.VectorData(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\global_settlement_points_v1.01.shp", encoding='latin')
    places.create_spatial_index()
    rivers = pg.VectorData("data/ne_10m_rivers_lake_centerlines.shp") 
    rivers.create_spatial_index()
    urban = pg.VectorData("data/ne_10m_urban_areas.shp") 
    urban.create_spatial_index()
    roads = pg.VectorData("data/ne_10m_roads.shp") 
    roads.create_spatial_index()

    # simulate
    i = 1

    # options
    n = 10
    extents = [10] + [40, 20] + [5, 1, 0.5, 0.1]
    quantities = [80, 40, 20, 10]
    distributions = ['random']
    alldatas = [
                    [], # no data layers
                    [
                    (rivers, {'fillcolor':(54,115,159), 'fillsize':0.08, 'legendoptions':{'title':'Rivers'}}), # three layers
                    (urban, {'fillcolor':(209,194,151), 'legendoptions':{'title':'Urban area'}}),
                    (roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),
                     ],
                ]
    projections = [None, # lat/lon
                   '+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', #'+init=EPSG:3857', # Web Mercator
                   '+proj=moll +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54009', # World Mollweide
                   '+proj=robin +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54030', # Robinson
                   ]
    resolutions = [2000, 1000, 500] #, 4000]
    imformats = ['png','jpg']
    textsizes = [8] #18
    metas = [{'title':'','legend':False,'arealabels':False}, # nothing
             {'title':'This is the Map Title','legend':True,'arealabels':True}, # meta boxes and area labels
             ]

    # region
    ci = 0
    while ci < n:
        center = (uniform(-160,160),uniform(-60,60))
        
        print('-------')
        print('CENTER:',center)

        # first render simple version
        print('SIMPLE')
        name = 'sim_{}'.format(i)

        extent = extents[0]
        quantity = quantities[0]
        distribution = distributions[0]
        datas = alldatas[0]
        projection = projections[0]
        resolution = resolutions[0]
        imformat = imformats[0]
        resolution = resolutions[0]
        textsize = textsizes[0]
        meta = metas[0]

        regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
        bbox = mapregion(**regionopts)
        if not valid_mapregion(bbox):
            print('!!! Not enough land area, skipping')
            continue
        
        placeopts = {'quantity':quantity, 'distribution':distribution}
        mapplaces = get_mapplaces(bbox, **placeopts)
        if len(mapplaces) < quantity:
            print('!!! Not enough places, skipping')
            continue

        ci += 1
        
        metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}
        textopts = {'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
        anchoropts = {'fillcolor':'black', 'fillsize':0.1}
        noiseopts = {'resolution':resolution, 'format':imformat}

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

        save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
        i += 1

        # then render complex version (alldata, allmeta)
        print('COMPLEX')
        name = 'sim_{}'.format(i)
        
        datas = alldatas[-1]
        meta = metas[-1]

        metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}
        textopts = {'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
        anchoropts = {'fillcolor':'black', 'fillsize':0.1}
        noiseopts = {'resolution':resolution, 'format':imformat}

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

        save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
        i += 1

        # (reset)
        datas = alldatas[0]
        meta = metas[0]
        metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}

        # then vary projections
        for projection in projections[1:]:
            print('PROJECTION',projection)
            name = 'sim_{}'.format(i)

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

            save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
            i += 1

        # (reset)
        projection = projections[0]

        # then vary resolutions + imformat
        for resolution,imformat in itertools.product(resolutions, imformats):
            print('RESOLUTION',resolution)
            print('IMFORMAT',imformat)
            name = 'sim_{}'.format(i)

            noiseopts = {'resolution':resolution, 'format':imformat}

            save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
            i += 1

        # (reset)
        resolution = resolutions[0]
        imformat = imformats[0]
        noiseopts = {'resolution':resolution, 'format':imformat}


        # then vary quantity
        for quantity in quantities[1:]:
            print('QUANTITY',quantity)
            name = 'sim_{}'.format(i)

            placeopts = {'quantity':quantity, 'distribution':distribution}
            mapplaces = get_mapplaces(bbox, **placeopts)

            metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}
            textopts = {'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
            anchoropts = {'fillcolor':'black', 'fillsize':0.1}
            noiseopts = {'resolution':resolution, 'format':imformat}

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

            save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
            i += 1

        # (reset)
        quantity = quantities[0]
        placeopts = {'quantity':quantity, 'distribution':distribution}
        mapplaces = get_mapplaces(bbox, **placeopts)

        # then vary distribution
        # ...

        # then vary extents
        for extent in extents[1:]:
            print('EXTENT:',extent)
            name = 'sim_{}'.format(i)

            regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
            bbox = mapregion(**regionopts)
            if not valid_mapregion(bbox):
                print('!!! Not enough land area, skipping')
                continue

            placeopts = {'quantity':quantity, 'distribution':distribution}
            mapplaces = get_mapplaces(bbox, **placeopts)
            if len(mapplaces) < 10:
                print('!!! Not Enough Places, Skipping')
                continue
            placeopts['quantity'] = len(mapplaces) # in case available places is less than desired
            
            metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}
            textopts = {'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
            anchoropts = {'fillcolor':'black', 'fillsize':0.1}
            noiseopts = {'resolution':resolution, 'format':imformat}

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

            save_map(name, mapp, mapplaces, resolution, regionopts, projection, anchoropts, textopts, metaopts, noiseopts)
            i += 1


        



