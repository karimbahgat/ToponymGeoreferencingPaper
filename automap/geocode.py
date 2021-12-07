
import sqlite3
import shapely, shapely.wkb


##class Matches(object):
##    def __init__(self, stream):
##        self.stream = stream
##        self.matches = []
##
##    def __bool__(self):
##        return next(self)
##
##    def __nonzero(self):
##        return self.__bool__()
##
##    def __next__(self):
##        nxt = next(self.stream, None)
##        if nxt:
##            self.matches.append(nxt)
##            return nxt
##
##    def __iter__(self):
##        for m in self.matches:
##            yield m
##            
##        nxt = next(self, None)
##        while nxt:
##            yield nxt
##            nxt = next(self, None)



class Online(object):
    def __init__(self):
        import geopy
        self.coder = geopy.geocoders.Nominatim()

    def geocode(self, name, limit=10):
        ms = self.coder.geocode(name, exactly_one=False, limit=limit) or []
        for m in ms:
            yield {'type': 'Feature',
                   'properties': {'name': m.address,
                                  },
                   'geometry': {'type': 'Point',
                                'coordinates': (m.longitude, m.latitude)
                                },
                   }


def wkb_to_shapely(wkbbuf):
    shp = shapely.wkb.loads(bytes(wkbbuf))
    return shp


class OptimizedCoder(object):
    def __init__(self, path=None):
        self.path = path or 'resources/gazetteers.db'
        self.db = sqlite3.connect(self.path)

    '''
    def geocode(self, name, limit=None):
        # NOT CORRRECT QUERY, RETURNS DUPLICATES
        if limit:
            raise NotImplemented("Geocode results 'limit' not yet implemented")
        results = self.db.cursor().execute("SELECT locs.data, locs.id, GROUP_CONCAT(names.name, '|'), locs.geom FROM locs, names, (SELECT data,id FROM names WHERE name = ? COLLATE NOCASE) AS m WHERE locs.id=m.id AND locs.data=m.data and names.id=m.id and names.data=m.data GROUP BY m.data,m.id", (name,))
        results = ({'type': 'Feature',
                   'properties': {'data':data,
                                  'id':ID,
                                  'name':names,
                                  'search':name,
                                  },
                   'geometry': wkb_to_shapely(geom).__geo_interface__,
                   } for data,ID,names,geom in results)
        return results #Matches(results)
    '''

    def geocode(self, name, limit=None):
        # NOT CORRRECT QUERY, RETURNS DUPLICATES
        if limit:
            raise NotImplemented("Geocode results 'limit' not yet implemented")
        #query = "SELECT locs.data, locs.id, GROUP_CONCAT(names.name, '|'), locs.geom FROM locs, names, (SELECT data,id FROM names WHERE name = ? COLLATE NOCASE) AS m WHERE locs.id=m.id AND locs.data=m.data and names.id=m.id and names.data=m.data GROUP BY m.data,m.id"
        _matches = "SELECT * FROM names WHERE name = ? COLLATE NOCASE"
        #_locmatches = "SELECT loc_id, GROUP_CONCAT(matches.name, '|') AS names FROM matches GROUP BY loc_id"
        _select = "sources.name,locs.loc_id,GROUP_CONCAT(names.name, '|'),locs.lon,locs.lat"
        #_from = "locmatches INNER JOIN locs ON locmatches.loc_id=locs.loc_id"
        _from = "sources,locs,names,matches WHERE names.loc_id=matches.loc_id AND names.loc_id=locs.loc_id AND locs.source_id=sources.source_id"
        _groupby = "matches.loc_id"
        query = "WITH matches AS ({_matches}) SELECT {_select} FROM {_from} GROUP BY {_groupby}".format(_matches=_matches, _select=_select, _from=_from, _groupby=_groupby)
        results = self.db.cursor().execute(query, (name,))
        results = ({'type': 'Feature',
                   'properties': {'data':data,
                                  'id':ID,
                                  'name':names,
                                  'search':name,
                                  },
                   'geometry': {'type':'Point', 'coordinates':[lon,lat]},
                   } for data,ID,names,lon,lat in results)
        return results #Matches(results)



class SQLiteCoder(object):
    def __init__(self, db=None, table=None):
        self.path = db
        self.db = sqlite3.connect(self.path)
        self.table = table

    def geocode(self, name, limit=10):
        #where = u"names like '%{0}%'".format(name)
        where = u" '|' || names || '|' like '%|{0}|%' ".format(name)
        #where = u"names like '{0}|%' or names like '%|{0}|%' or names like '%|{0}'".format(name)
        results = self.db.cursor.execute('select names,lon,lat from {table} where {where} limit {limit}'.format(table=self.table, where=where, limit=limit))
        results = ({'type': 'Feature',
                   'properties': {'name': names,
                                  },
                   'geometry': {'type': 'Point',
                                'coordinates': (lon,lat)
                                },
                   } for names,lon,lat in results)
        return results #Matches(results)


class GNS(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\BIGDATA\gazetteer data\prepped\gns.db'
    table = 'data'


class GeoNames(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\BIGDATA\gazetteer data\prepped\geonames.db'
    table = 'data'


class OSM(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\BIGDATA\gazetteer data\prepped\osm.db'
    table = 'data'


class CIESIN(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\BIGDATA\gazetteer data\prepped\ciesin.db'
    table = 'data'


class NatEarth(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\BIGDATA\gazetteer data\prepped\natearth.db'
    table = 'data'

