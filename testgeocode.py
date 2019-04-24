
from automap import geocode
from time import time


def view(results, text=True):
    import pythongis as pg
    # setup map
    m = pg.renderer.Map()
    m.add_layer(r"C:\Users\kimok\Desktop\gazetteer data\raw\ne_10m_admin_0_countries.shp", fillcolor='gray')
    # options
    kwargs = {}
    if text:
        kwargs.update(text=lambda f: f['names'][:20], textoptions=dict(textsize=3))
    # add
    d = pg.VectorData(fields=['names'])
    for match in results:
        name = match['properties']['name']
        geoj = match['geometry']
        d.add_feature([name], geoj)
    m.add_layer(d, fillcolor='blue', **kwargs)
    # view
    m.view()


coder = geocode.GeoNames()

for name in 'new york|paris|oslo|tokyo|williamsburg'.split('|'):
    print name
    t = time()
    res = coder.geocode(name, limit=None)
    #print len(res)
    print time()-t
    view(res)
