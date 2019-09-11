
import pythongis as pg

from random import uniform



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



# sample n places within focus region
def _sampleplaces(bbox, n, distribution):
    '''
    Samples...
    '''
    radius = 0.1
    i = 0
    print 'target box',bbox
    while True:
        if distribution == 'random':
            x = uniform(bbox[0], bbox[2])
            y = uniform(bbox[1], bbox[3])
        else:
            raise Exception('Distribution type {} is not a valid option'.format(distribution))
        bufbox = [x-radius, y-radius, x+radius, y+radius]
        for r in places.quick_overlap(bufbox): #intersection('geom', bufbox):
            #print '---'
            #print x,y
            #print bufbox
            #print r.bbox
            i += 1
            yield r
            # yield only first match inside bbox, then break
            break
        if i >= n-1:
            break

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
        name = f['NAME'] #r['names'].split('|')[0]
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
        arealabels = {'text':lambda f: f['NAME'].upper(), 'textoptions': {'textsize':textopts['textsize']*1.5}}
    else:
        arealabels = {}
    m.add_layer(countries, fillcolor=(255,222,173), outlinewidth=0.2, outlinecolor=(100,100,100),
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
def save_map(name, mapp, mapplaces, noiseopts):
    
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






####################
# RUN

if __name__ == '__main__':
    import itertools
    
    # load data
    countries = pg.VectorData("data/ne_10m_admin_0_countries.shp")
    places = pg.VectorData("data/ne_10m_populated_places.shp") 
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
    centers = [(-116,40)] #(uniform(-160,160),uniform(-60,60)) for _ in range(n)]
    extents = [40, 20, 5, 1]
    quantities = [10, 20, 40, 80] 
    distributions = ['random']
    alldatas = [    
                   [
                    (rivers, {'fillcolor':(54,115,159), 'fillsize':0.08, 'legendoptions':{'title':'Rivers'}}), # three layers
                    (urban, {'fillcolor':(209,194,151), 'legendoptions':{'title':'Urban area'}}),
                    (roads, {'fillcolor':(187,0,0), 'fillsize':0.08, 'legendoptions':{'title':'Roads'}}),
                     ],
                   [], # no data layers
                ]
    projections = [#'+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', #'+init=EPSG:3857', # Web Mercator
                   #'+init=ESRI:54009', # World Mollweide
                   #'+init=ESRI:54030', # Robinson
                   None, # lat/lon
                   ]
    resolutions = [2000, 1000, 500] #, 4000]
    imformats = ['jpg','png']
    noises = [] # ADD RANDOM PIXEL NOISE TO IMAGE ?? 
    textsizes = [18-8]
    metas = [{'title':'This is the Map Title','legend':True,'arealabels':False}, # meta boxes
             {'title':'','legend':False,'arealabels':True}, # area labels
             {'title':'','legend':False,'arealabels':False}] # nothing

##    import itertools
##    combis = list(itertools.product(centers,extents,quantities,distributions,alldatas,projections,resolutions,imformats,textsizes,metas))
##    print(len(combis), 'permutations')
##    fsddf

    # region
    for center,extent in itertools.product(centers, extents):
        regionopts = {'center':center, 'extent':extent, 'aspect':0.70744225834}
        bbox = mapregion(**regionopts)

        # places
        for quantity,distribution in itertools.product(quantities,distributions):
            #placeopts = {'quantity':40, 'distribution':'random'}
            placeopts = {'quantity':quantity, 'distribution':distribution}
            mapplaces = get_mapplaces(bbox, **placeopts)

            # data noise
            for datas in alldatas:

                # projections
                # UGLY...
                for projection in projections:
                    
##                    projdatas = []
##                    for datadef in datas:
##                        if datadef:
##                            data,style = datadef
##                            if projection:
##                                data.crs = '+init=EPSG:4326' # WGS84 unprojected
##                                data = data.manage.reproject(projection)
##                            projdatas.append((data,style))
##                    if projection:
##                        countries.crs = '+init=EPSG:4326' # WGS84 unprojected
##                        projcountries = countries.manage.reproject(projection)
##                        mapplaces.crs = '+init=EPSG:4326' # WGS84 unprojected
##                        projmapplaces = mapplaces.manage.reproject(projection)
##                    else:
##                        projcountries = countries
##                        projmapplaces = mapplaces

                    # metadata
                    for meta in metas:
                        metaopts = {'title':meta['title'], 'titleoptions':{'fillcolor':'white'}, 'legend':meta['legend'], 'arealabels':meta['arealabels']}

                        # textsizes
                        for textsize in textsizes:
                            textopts = {'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0}
                            anchoropts = {'fillcolor':'black', 'fillsize':0.1}

                            # render
                            mapp = render_map(bbox,
                                             mapplaces,
                                             datas,
                                             resolutions[0],
                                             regionopts=regionopts,
                                             projection=projection,
                                             anchoropts=anchoropts,
                                             textopts=textopts,
                                             metaopts=metaopts,
                                             )

                            # ADD GAZETTEER DISTORTION PARAM

                            # resolutions
                            for resolution,imformat in itertools.product(resolutions, imformats):

                                name = 'sim_{}'.format(i)
                                print(name)
                                for opts in [regionopts,placeopts,projection,datas,textopts,metaopts]:
                                    print(opts)

                                noiseopts = {'resolution':resolution, 'format':imformat}

                                save_map(name, mapp, mapplaces, noiseopts)
                                
                                i += 1

    




