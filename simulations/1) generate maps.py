
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
def render_map(name, bbox, mapplaces, datas, regionopts, qualityopts, anchoropts, textopts):
    # determine resolution
    width = 4000
    height = int(width * regionopts['aspect'])
    
    # render pure map image
    m = pg.renderer.Map(width, height,
                        background=(91,181,200))
    m.add_layer(countries, fillcolor=(255,222,173), outlinewidth=0.18)

    for data,style in datas:
        m.add_layer(data, **style)
        
    m.add_layer(mapplaces,
                text=lambda f: f['name'],
                textoptions=textopts,
                **anchoropts)
    m.zoom_bbox(*bbox)

    # downscale to resolution
    ratio = qualityopts['resolution'] / float(width)
    newwidth = qualityopts['resolution']
    newheight = int(height * ratio)

    # save
    imformat = qualityopts['format']
    m.save('maps/{}_image.{}'.format(name, imformat))

    m.img.resize((newwidth, newheight), 1).save('maps/{}_image.{}'.format(name, imformat))

    # store rendering with original geo coordinates
    affine = m.drawer.coordspace_transform
    r = pg.RasterData('maps/{}_image.{}'.format(name, imformat)) 
    r.set_geotransform(affine=m.drawer.coordspace_invtransform)
    r.save('maps/{}_truth.tif'.format(name))

    # store the original place coordinates
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

    return m







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

    # region
    for _region_ in [None]:
        regionopts = {'center':(10,10), 'extent':20.0, 'aspect':0.70744225834}
        bbox = mapregion(**regionopts)

        # places
        for placeoptvals in itertools.product([10, 20, 40, 80], ['random']):
            #placeopts = {'quantity':40, 'distribution':'random'}
            placeopts = dict(zip(['quantity','distribution'], placeoptvals))
            mapplaces = get_mapplaces(bbox, **placeopts)

            # data noise
            for datas in [[ (rivers, {'fillcolor':(54,115,159), 'fillsize':0.08}),
                            (urban, {'fillcolor':(209,194,151)}),
                            (roads, {'fillcolor':(187,0,0), 'fillsize':0.08}),
                             ]]:

                # apply projection
                # ...

                # resolutions
                for resolution in [500, 1000, 2000, 4000]:

                    # ALSO DO
                    # - metadata noise: ie big title + notebox
                    # - ...

                    # textsizes
                    for textsize in [14, 18, 22]:
                        name = 'sim_{}'.format(i)
                        print(name)

                        # render
                        render_map(name,
                                     bbox,
                                     mapplaces,
                                     datas,
                                     regionopts=regionopts,
                                     qualityopts={'resolution':resolution, 'format':'gif'},
                                     anchoropts={'fillcolor':'black', 'fillsize':0.1},
                                     textopts={'textsize':textsize, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0},
                                     )
                        
                        i += 1

    




