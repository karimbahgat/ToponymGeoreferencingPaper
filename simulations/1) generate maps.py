
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
            print '---'
            print x,y
            print bufbox
            print r.bbox
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
        print f
        name = f['NAME'] #r['names'].split('|')[0]
        row = [name]
        mapplaces.add_feature(row, f.geometry) #r['geom'].__geo_interface__)
            
    return mapplaces



# render map
def render_map(name, bbox, mapplaces, regionopts, qualityopts, anchoropts, textopts):
    # determine resolution
    pixelwidth = qualityopts['resolution']
    pixelheight = int(pixelwidth * regionopts['aspect'])
    
    # render pure map image
    m = pg.renderer.Map(pixelwidth, pixelheight, background='white')
    m.add_layer(countries, fillcolor=(255,222,173))
    m.add_layer(mapplaces,
                text=lambda f: f['name'],
                textoptions=textopts,
                **anchoropts)
    m.zoom_bbox(*bbox)

    # save
    imformat = qualityopts['format']
    m.save('maps/{}_image.{}'.format(name, imformat))

    # store rendering with original geo coordinates
    affine = m.drawer.coordspace_transform
    r = pg.RasterData('maps/{}_image.{}'.format(name, imformat)) 
    r.set_geotransform(affine=m.drawer.coordspace_invtransform)
    r.save('maps/{}_truth.tif'.format(name))

    return m




####################
# MAIN

def simulate_map(prefix, regionopts, qualityopts, projopts, placeopts, anchoropts, textopts):
    name = prefix #formatname(prefix)

    # region
    bbox = mapregion(**regionopts)

    # places
    mapplaces = get_mapplaces(bbox, **placeopts)
    # apply projection...

    # render
    mapp = render_map(name, bbox, mapplaces, regionopts, qualityopts, anchoropts, textopts)

    # store the original place coordinates
    mapplaces.add_field('col')
    mapplaces.add_field('row')
    mapplaces.add_field('x')
    mapplaces.add_field('y')
    for f in mapplaces:
        x,y = f.geometry['coordinates']
        col,row = mapp.drawer.coord2pixel(x,y)
        f['col'] = col
        f['row'] = row
        f['x'] = x
        f['y'] = y
    mapplaces.save('maps/{}_placenames.geojson'.format(name))



####################
# RUN

if __name__ == '__main__':
    # load data
    countries = pg.VectorData("data/ne_10m_admin_0_countries.shp")
    places = pg.VectorData("data/ne_10m_populated_places.shp") 
    places.create_spatial_index()

    # simulate
    simulate_map('test',
                 regionopts={'center':(10,10), 'extent':20.0, 'aspect':0.5},
                 qualityopts={'resolution':500, 'format':'gif'},
                 projopts=None,
                 placeopts={'quantity':40, 'distribution':'random'},
                 anchoropts={'fillcolor':'black', 'fillsize':0.1},
                 textopts={'textsize':6, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0},
                 )

    




