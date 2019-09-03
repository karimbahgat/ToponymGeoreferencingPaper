
import automap as mapfit
import pythongis as pg
import geostream as gs

from random import uniform

# params
TESTCOUNTRY = 'Burkina Faso'
PLACENAMES = 50

# load data
countries = pg.VectorData(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_admin_0_countries.shp")
places = pg.VectorData(r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\raw\ne_10m_populated_places.shp") #gs.Table(r"C:\Users\kimok\Desktop\gazetteer data\prepped\gns.db", 'data', 'w')
places.create_spatial_index()

def sampleplaces(bbox, n, radius=0.1):
    '''
    Samples...
    '''
    i = 0
    print 'target box',bbox
    while True:
        x = uniform(bbox[0], bbox[2])
        y = uniform(bbox[1], bbox[3])
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

###########################

# select map region by zooming to a country
co = countries.select(lambda f: f['GEOUNIT'] == TESTCOUNTRY)
bbox = co.bbox

# sample n places within focus region
coplaces = pg.VectorData()
coplaces.fields = ['name']
for f in sampleplaces(bbox, PLACENAMES):
    print f
    name = f['NAME'] #r['names'].split('|')[0]
    row = [name]
    coplaces.add_feature(row, f.geometry) #r['geom'].__geo_interface__)

# render
if 1:
    # render map
    m = pg.renderer.Map(2000, 2000, background='white')
    m.add_layer(co, fillcolor=(255,222,173))
    m.add_layer(coplaces,
                fillsize=0.1,
                fillcolor='black',
                text=lambda f: f['name'],
                textoptions={'textsize':6, 'anchor':'sw', 'xoffset':0.5, 'yoffset':0})
    m.zoom_bbox(*bbox)
    m.save('maps/test.png')

    # store rendering with original geo coordinates
    affine = m.drawer.coordspace_transform
    r = pg.RasterData('maps/test.png') 
    r.set_geotransform(affine=m.drawer.coordspace_invtransform)
    r.save('maps/test.tif')

    # save original place coordinates used in rendering
    coplaces.add_field('col')
    coplaces.add_field('row')
    coplaces.add_field('x')
    coplaces.add_field('y')
    for f in coplaces:
        x,y = f.geometry['coordinates']
        col,row = m.drawer.coord2pixel(x,y)
        f['col'] = col
        f['row'] = row
        f['x'] = x
        f['y'] = y
    coplaces.save('maps/test_placenames.geojson')


