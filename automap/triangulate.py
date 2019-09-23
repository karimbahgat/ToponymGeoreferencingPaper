"""
"""

import itertools

from . import shapematch


def triangulate(coder, names, positions, matchcandidates=None):
    assert len(names) == len(positions)
    assert len(names) >= 3

    # define input names/positions as a polygon feature
    findpoly = {'type': 'Feature',
               'properties': {'combination': names,
                              },
               'geometry': {'type': 'Polygon',
                            'coordinates': [positions]
                            },
               }

    # find matches for each name
    #print 'finding matches', names
    if matchcandidates:
        assert len(matchcandidates) == len(names)
    else:
        matchcandidates = []
        for name in names:
            match = list(coder.geocode(name))
            if match:
                matchcandidates.append(match)

    if len(matchcandidates) < 3:
        return []
            
    #for match in matchcandidates:
    #    print len(match)

    # find unique combinations of all possible candidates
    #print 'combining'
    combis = list(itertools.product(*matchcandidates))
    combipolys = []
    for combi in combis:
        #print '--->', combi
        # make into polygon feature
        combinames = [f['properties']['name'] for f in combi]
        combipositions = [f['geometry']['coordinates'] for f in combi]
        combipoly = {'type': 'Feature',
                   'properties': {'combination': combinames,
                                  },
                   'geometry': {'type': 'Polygon',
                                'coordinates': [combipositions]
                                },
                   }
        combipolys.append(combipoly)

    # prep/normalize combipolys
    #print 'prepping'
    combipolys = shapematch.prep_pool(combipolys)

    # find closest match
    #print 'finding'
    matches = shapematch.find_exact_match_prepped(findpoly, combipolys)
    return matches

##def triangulate_add(existing, add):
##    
##    fjdkslfjldsjfkljsdkjfklsj
##    
##    names,positions = existing
##    addname,addpos = add
##
##    # define input names/positions as a polygon feature
##    findpoly = {'type': 'Feature',
##               'properties': {'combination': list(names) + [addname],
##                              },
##               'geometry': {'type': 'Polygon',
##                            'coordinates': [positions + [addpos]]
##                            },
##               }
##
##    # find candidates for the added name
##    matches = geocode(name)
##
##    # create possible polygons after adding
##    combipolys = []
##    for m in matches:
##        #print '--->', combi
##        # make into polygon feature
##        combinames = [f['properties']['name'] for f in combi]
##        combipositions = [f['geometry']['coordinates'] for f in combi]
##        combipoly = {'type': 'Feature',
##                   'properties': {'combination': list(names) + [m.address],
##                                  },
##                   'geometry': {'type': 'Polygon',
##                                'coordinates': [positions + [(m.longitude,m.latitude)]]
##                                },
##                   }
##        combipolys.append(combipoly)
##
##    
##    added = {'type': 'Feature',
##               'properties': {'combination': list(names) + [addname],
##                              },
##               'geometry': {'type': 'Polygon',
##                            'coordinates': [positions + [addpos]]
##                            },
##               }
##
##    added = shapematch.prep_pool(added)
##    matches = shapematch.find_exact_match_prepped(findpoly, combipolys)
##    return matches[0]

def triangulate_add(coder, origs, matches, add, addcandidates=None):
    orignames, origpositions = zip(*origs)
    matchnames, matchpositions = zip(*matches)

    addname,addpos = add

    # define input names/positions as a polygon feature
    findpoly = {'type': 'Feature',
               'properties': {'combination': list(orignames) + [addname],
                              },
               'geometry': {'type': 'Polygon',
                            'coordinates': [list(origpositions) + [addpos]]
                            },
               }

    # find matches for each name
    #print 'finding matches', addname
    if addcandidates:
        match = addcandidates
    else:
        match = list(coder.geocode(addname))
        if not match:
            return False
            
    #print len(match)

    # find unique combinations of all possible candidates
    #print 'combining'
    combipolys = []
    for m in match:
        #print '--->', combi
        # make into polygon feature
        combinames = list(matchnames) + [m['properties']['name']]
        combipositions = list(matchpositions) + [m['geometry']['coordinates']]
        combipoly = {'type': 'Feature',
                   'properties': {'combination': combinames,
                                  },
                   'geometry': {'type': 'Polygon',
                                'coordinates': [combipositions]
                                },
                   }
        combipolys.append(combipoly)

    # prep/normalize combipolys
    #print 'prepping'
    combipolys = shapematch.prep_pool(combipolys)

    # find closest match
    #print 'finding'
    matches = shapematch.find_exact_match_prepped(findpoly, combipolys)
    return matches




        
