
import geostream as gs
import geopy


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




class SQLiteCoder(object):
    def __init__(self, db=None, table=None):
        if db: self.db = db
        if table: self.table = table
        self.stream = gs.Table(self.db, self.table)

    def geocode(self, name, limit=10):
        #where = u"names like '%{0}%'".format(name)
        where = u" '|' || names || '|' like '%|{0}|%' ".format(name)
        #where = u"names like '{0}|%' or names like '%|{0}|%' or names like '%|{0}'".format(name)
        results = self.stream.select(['names','lon','lat'],
                                     where=where,
                                     limit=limit)
        results = ({'type': 'Feature',
                   'properties': {'name': names,
                                  },
                   'geometry': {'type': 'Point',
                                'coordinates': (lon,lat)
                                },
                   } for names,lon,lat in results)
        return results #Matches(results)


class GNS(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\gazetteer data\prepped\gns.db'
    table = 'data'


class GeoNames(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\gazetteer data\prepped\geonames.db'
    table = 'data'


class OSM(SQLiteCoder):
    db = r'C:\Users\kimok\Desktop\gazetteer data\prepped\osm.db'
    table = 'data'


