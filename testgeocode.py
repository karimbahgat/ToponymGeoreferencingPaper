
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


##import geostream as gs
##for d in 'un natearth ciesin osm geonames gns'.split():
##    print d
##    t = gs.Table(r'C:\Users\kimok\Desktop\gazetteer data\prepped\{}.db'.format(d), 'data', 'w')
##    t.create_index('names', replace=True) #, nocase=True)
##fdsfsd

coder = geocode.OptimizedCoder()

#coder.stream.workspace.db.cursor().execute('PRAGMA temp_store = MEMORY') 
#coder.stream.workspace.db.cursor().execute('PRAGMA cache_size = 10000000')
#coder.stream.workspace.db.cursor().execute('PRAGMA mmap_size = 10000000')
#coder.stream.workspace.db.cursor().execute('PRAGMA journal_mode = MEMORY')
#coder.stream.workspace.db.cursor().execute('PRAGMA synchronous = 0')
#print coder.stream
#print list(coder.stream.workspace.db.cursor().execute('PRAGMA index_list(data)'))
#print list(coder.stream.workspace.db.cursor().execute("SELECT * FROM SQLite_master WHERE type = 'index' AND tbl_name = 'data'"))
#fsdfs

#coder.stream.workspace.db.cursor().execute('PRAGMA case_sensitive_like = ON')
for name in 'Quargla|new york|paris|oslo|tokyo|williamsburg'.split('|'):
    print name
    t = time()
    res = list(coder.geocode(name.title(), limit=None))
    #res = list(coder.stream.workspace.db.cursor().execute("select * from data where names like '%|{0}|%'".format(name)))
    #res = list(coder.stream.workspace.db.cursor().execute("select * from data where names like '{0}' or names like '{0}|%' or names like '%|{0}|%' or names like '%|{0}'".format(name)))
    #print len(res)
    print time()-t
    #for r in res:
    #    print r['properties']
    view(res)
