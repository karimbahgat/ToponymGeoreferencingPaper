# OLD VERSION, NO LONGER USED
# see instead patternmatch.py

import shapely, shapely.geometry

def normalize(geoj):
    if 'Multi' in geoj['type']:
        polys = geoj['coordinates']
    else:
        polys = [geoj['coordinates']]

    xs,ys = [],[]
    for poly in polys:
        ext = poly[0] # exterior only
        _xs,_ys = zip(*ext)
        xs += _xs
        ys += _ys
    xmin,ymin = min(xs),min(ys)
    xmax,ymax = max(xs),max(ys)
    w,h = xmax-xmin, ymax-ymin
    scale = max(w,h)

    outpolys = []
    for poly in polys:
        
        ext = poly[0] # exterior only
        xs,ys = zip(*ext)
        xs = [(x-xmin)/float(scale) for x in xs]
        ys = [(y-ymin)/float(scale) for y in ys]
        ext = zip(xs,ys)
        
        poly = [ext] # no holes
        outpolys.append(poly)

    if 'Multi' in geoj['type']:
        coords = outpolys
    else:
        coords = outpolys[0]

    return {'type':geoj['type'], 'coordinates':coords}

def shapediff(geoj1, geoj2):
    norm2 = normalize(geoj2)
    g2 = shapely.geometry.asShape(norm2)
    
    if 'Multi' in g2.geom_type:
        lines = [poly.boundary for poly in g2.geoms]
        g2 = shapely.geometry.MultiLineString(lines)
    else:
        g2 = g2.boundary
        
    dists = []

    norm1 = normalize(geoj1)
    norm1 = shapely.geometry.asShape(norm1).simplify(0.1).__geo_interface__
    if 'Multi' in geoj1['type']:
        polys = norm1['coordinates']
    else:
        polys = [norm1['coordinates']]

    # point-wise
    for poly in polys:
        ext = poly[0] # exterior only
        for p in ext: 
            g1 = shapely.geometry.Point(p)
            d = g1.distance(g2)
            dists.append(d)

    avg = sum(dists)/len(dists)
    return avg

def shapediff_prepped(polys, g2):       
    dists = []

    # point-wise
    for poly in polys:
        ext = poly[0] # exterior only
        for p in ext: 
            g1 = shapely.geometry.Point(p)
            d = g1.distance(g2)
            dists.append(d)

    avg = sum(dists)/len(dists)
    #med = sorted(dists)[int(len(dists)/2.0)]
    return avg,dists

def exactdiff_prepped(polys, geom):       
    dists = []

    # point-wise
    for poly in polys:
        ext = poly[0] # exterior only
        for p1,p2 in zip(ext, geom.coords):
            g1 = shapely.geometry.Point(p1)
            g2 = shapely.geometry.Point(p2)
            d = g1.distance(g2)
            dists.append(d)

    avg = sum(dists)/len(dists)
    #med = sorted(dists)[int(len(dists)/2.0)]
    return avg,dists

def find_match(test, pool):
    from time import time
    # prep pool
    t=time()
    g2s = []
    for f in pool:
        norm2 = normalize(f['geometry'])
        g2 = shapely.geometry.asShape(norm2)
        
        if 'Multi' in g2.geom_type:
            lines = [poly.boundary for poly in g2.geoms]
            g2 = shapely.geometry.MultiLineString(lines)
        else:
            g2 = g2.boundary
        g2s.append( (f,g2) )
    #print time()-t

    # prep test
    t=time()
    norm1 = normalize(test)
    norm1 = shapely.geometry.asShape(norm1).simplify(0.1).__geo_interface__
    if 'Multi' in test['type']:
        polys = norm1['coordinates']
    else:
        polys = [norm1['coordinates']]
    #print time()-t

    # calculate diffs
    results = []
    for f,g2 in g2s:
        #name = f['properties']['GEOUNIT']
        #print name
        diff,diffs = shapediff_prepped(polys, g2)
        #print diff

        results.append( (f,diff,diffs) )

    # return best
    return sorted(results, key=lambda pair: pair[1])

def find_match_prepped(test, pool):
    from time import time
    
    # prep test
    t=time()
    norm1 = normalize(test['geometry'])
    norm1 = shapely.geometry.asShape(norm1).simplify(0.1).__geo_interface__
    if 'Multi' in test['type']:
        polys = norm1['coordinates']
    else:
        polys = [norm1['coordinates']]
    #print time()-t

    # calculate diffs
    t=time()
    results = []
    for f,g2 in pool:
        #name = f['properties']['GEOUNIT']
        #print name
        diff,diffs = shapediff_prepped(polys, g2)
        #print diff

        results.append( (f,diff,diffs) )
    #print time()-t

    # return best
    return sorted(results, key=lambda pair: pair[1])

def find_exact_match_prepped(test, pool):
    from time import time
    
    # prep test
    t=time()
    norm1 = normalize(test['geometry'])
    #norm1 = shapely.geometry.asShape(norm1).simplify(0.1).__geo_interface__
    if 'Multi' in test['type']:
        polys = norm1['coordinates']
    else:
        polys = [norm1['coordinates']]
    #print time()-t

    # calculate diffs
    t=time()
    results = []
    for f,g2 in pool:
        #name = f['properties']['GEOUNIT']
        #print name
        diff,diffs = exactdiff_prepped(polys, g2)
        #print diff

        results.append( (f,diff,diffs) )
    #print time()-t

    # return best
    return sorted(results, key=lambda pair: pair[1])

def prep_pool(pool):
    from time import time
    # prep pool
    t=time()
    g2s = []
    for f in pool:
        norm2 = normalize(f['geometry'])
        g2 = shapely.geometry.asShape(norm2)
        
        if 'Multi' in g2.geom_type:
            lines = [poly.boundary for poly in g2.geoms]
            g2 = shapely.geometry.MultiLineString(lines)
        else:
            g2 = g2.boundary
        g2s.append( (f,g2) )
    #print time()-t
    return g2s
    



##if __name__ == '__main__':
##    import pythongis as pg
##    from time import time
##    import cProfile
##
##    # selftest
####    data = pg.VectorData(r"C:\Users\kimok\Downloads\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp")
####    test = data.select(lambda f: f['GEOUNIT']=='India').manage.clean(0.3)
####    test = list(test)[0].geometry
####
####    feat,diff = find_country(test)
####    print feat.row, diff
####    feat.view()
##
##    # handdrawn test
##    test = {"type": "Polygon",
##            "coordinates": [
##          [
##            [
##              11.25,
##              57.51582286553883
##            ],
##            [
##              11.6015625,
##              59.88893689676585
##            ],
##            [
##              17.578125,
##              67.47492238478702
##            ],
##            [
##              31.289062500000004,
##              69.41124235697256
##            ],
##            [
##              28.4765625,
##              71.52490903732816
##            ],
##            [
##              13.7109375,
##              69.77895177646761
##            ],
##            [
##              2.4609375,
##              60.06484046010452
##            ],
##            [
##              4.5703125,
##              55.97379820507658
##            ],
##            [
##              11.25,
##              57.51582286553883
##            ]
##          ]
##        ]}
##    pool = pg.VectorData('countries_simple.geojson')
##    pool = prep_pool(pool)
##    matches = find_match_prepped(test, pool)
##    print [m['GEOUNIT'] for m,d,ds in matches[:3]]
##    fdsfs
##
##    # sourcetest
##    #pool = pg.VectorData(r"C:\Users\kimok\Downloads\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp")
##    #pool = pool.manage.clean(0.01)
##    #pool.save('countries_simple.geojson')
##    pool = pg.VectorData('countries_simple.geojson')
##    #pool.view()
##    pool = prep_pool(pool)
##    
##    tests = pg.VectorData(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp")
##    for i,test in enumerate(tests):
##        print i,'matching...'
##        t=time()
##        matches = find_match_prepped(test.geometry, pool)
##        print time()-t
##        match,diff,diffs = matches[0]
##        print test['CNTRY_NAME'],'--->',match['GEOUNIT'],diff
##        print [(m['GEOUNIT'],d) for m,d,ds in matches[1:4]]
##        
####        if 'Indonesia' in test['CNTRY_NAME']:
####            #print sorted(diffs)
####            for m,d,ds in matches:
####                #print m['GEOUNIT'],d
####                if 'Indonesia' in m['GEOUNIT']:
####                    out = pg.VectorData()
####                    out.add_feature([], normalize(test.geometry))
####                    out.add_feature([], normalize(m.geometry))
####                    out.view(fillcolor=None)
####            out = pg.VectorData()
####            out.add_feature([], normalize(test.geometry))
####            out.add_feature([], normalize(match.geometry))
####            out.view(fillcolor=None)
####
####        if 0:
####            out = pg.VectorData()
####            out.add_feature([], normalize(test.geometry))
####            out.add_feature([], normalize(match.geometry))
####            out.view(fillcolor=None)
##
##        print '----'


    


