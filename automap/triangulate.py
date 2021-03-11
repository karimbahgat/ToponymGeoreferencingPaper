"""
"""

import itertools

from . import shapematch
from . import geocode


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

def triang(coder, test, matchcandidates=None):
    # TODO: maybe superfluous, maybe just integrate right into "triangulate"?? 
    names,positions = zip(*test)
    # reverse ys due to flipped image coordsys
    maxy = max((y for x,y in positions))
    positions = [(x,maxy-y) for x,y in positions]
    # triangulate
    #print 99,names,positions
    matches = triangulate(coder, names, positions, matchcandidates)
    #for f,diff,diffs in matches[:1]:
        #print 'error:', round(diff,6)
        #for c in f['properties']['combination']:
        #    print c
        #viewmatch(positions, f)
    return matches

def find_matches(test, thresh=0.25, minpoints=8, mintrials=8, maxiter=500, maxcandidates=None, n_combi=3, db=None, source='best', debug=False):
    # filter to those that can be geocoded
    print 'geocode and filter'
    coder = geocode.OptimizedCoder(db)

    if source == 'best' or source == 'avg':
        mintrials = 30
        maxiter = 10000
    
    import time
    testres = []
    for nxtname,nxtpos in test:
        print 'geocoding',nxtname
        try:
            res = list(coder.geocode(nxtname, maxcandidates))
            if res:
                if source == 'avg':
                    import math
                    resnew = []
                    for r in res:
                        mayberes = [r2 for r2 in res if r != r2 and r['properties']['data'] != r2['properties']['data']]
                        x,y = r['geometry']['coordinates']
                        mayberes = [r2 for r2 in mayberes if math.hypot(abs(r2['geometry']['coordinates'][0]-x), abs(r2['geometry']['coordinates'][1]-y)) < 0.5]
                        if mayberes:
                            xs,ys = zip(*[r2['geometry']['coordinates'] for r2 in mayberes])
                            xm = sum(xs)/float(len(xs))
                            ym = sum(ys)/float(len(ys))
                            r['geometry'] = dict(type='Point', coordinates=(xm,ym))
                        resnew.append(r)
                    res = resnew
                elif source == 'best':
                    pass # just keep all the results and choose best matching ones
                else:
                    res = [r for r in res if r['properties']['data']==source]
                if res:
                    testres.append((nxtname,nxtpos,res))
                #time.sleep(0.1)
        except Exception as err:
            print 'EXCEPTION:', err
        
    #testres = [(nxtname,nxtpos,res)
    #           for nxtname,nxtpos,res in testres if res and len(res)<10]
    #testres = [(nxtname,nxtpos,res[:maxcandidates])
    #           for nxtname,nxtpos,res in testres]

    # print names to be tested
    for nxtname,nxtpos,res in testres:
        print nxtname,len(res)

    ### TEST: exhaustive search (experimental, not yet working)
##    n_combi = len(testres)
##    combis = list(itertools.combinations(testres, n_combi))
##    print '\n'+'finding all possible triangles of {} possible combinations'.format(len(combis))
##    triangles = []
##    for i,tri in enumerate(combis):
##        print '-----'
##        print 'try triangle %s of %s' % (i, len(combis))
##        try: best = triang([tr[:2] for tr in tri],
##                           matchcandidates=[tr[2] for tr in tri])
##        except Exception as err: print 'EXCEPTION RAISED:',err
##        if best:
##            f,diff,diffs = best[0]
##            valid = [tr[:2] for tr in tri]
##            triangles.append((valid, f, diff))
    ### END TEST

    # find all triangles from all possible 3-point combinations
    combis = itertools.combinations(testres, n_combi)
    # sort randomly to avoid local minima
    from random import uniform
    combis = sorted(combis, key=lambda x: uniform(0,1))
    # sort by length of possible geocodings, ie try most unique first --> faster+accurate
    combis = sorted(combis, key=lambda gr: sum((len(res) for nxtname,nxtpos,res in gr)))

    print '\n'+'finding all possible triangles of {} possible combinations'.format(len(combis))
    triangles = []
    for i,tri in enumerate(combis):
        if debug:
            print '-----'
            print 'try triangle %s of %s' % (i, len(combis))
            print '\n'.join([repr((tr[0],len(tr[2]))) for tr in tri])
        # try triang
        best = None
        try: best = triang(coder,
                           [tr[:2] for tr in tri],
                           matchcandidates=[tr[2] for tr in tri])
        except Exception as err: print 'EXCEPTION RAISED:',err
        if best:
            f,diff,diffs = best[0]
            #print f
            if debug:
                print 'error:', round(diff,6)
            if diff < thresh:
                if debug:
                    print 'TRIANGLE FOUND'
                valid = [tr[:2] for tr in tri]

                # ...
                for nxtname,nxtpos,res in testres:
                    if debug:
                        print 'trying to add incrementally:',nxtname,nxtpos
                    orignames,origcoords = zip(*valid)
                    orignames,origcoords = list(orignames),list(origcoords)
                    matchnames = list(f['properties']['combination'])
                    matchcoords = list(f['geometry']['coordinates'][0])
                    
                    if nxtpos in origcoords: continue
                    maxy = max((y for x,y in origcoords))
                    maxy = max(maxy,nxtpos[1])
                    nxtposflip = (nxtpos[0],maxy-nxtpos[1])
                    origcoordsflip = [(x,maxy-y) for x,y in origcoords + [nxtpos]]
                    best = triangulate_add(coder,
                                           zip(orignames,origcoordsflip),
                                           zip(matchnames,matchcoords),
                                           (nxtname,nxtposflip),
                                           res)
                    if not best: continue
                    mf,mdiff,mdiffs = best[0]
                    if mdiff < thresh:
                        if debug:
                            print 'ADDING'
                        valid.append((nxtname,nxtpos))
                        f = mf
                
                print '\n'+'MATCHES FOUND (error=%r)' % round(diff,6)
                print '>>>', repr([n for n,p in valid]),'-->',[n[:15] for n in f['properties']['combination']]

                triangles.append((valid,f,diff))
                
        if debug:
            print '%s triangles so far:' % len(triangles)
        
        #print '\n>>>'.join([repr((round(tr[2],6),[n for n,p in tr[0]],'-->',[n[:15] for n in tr[1]['properties']['combination']]))
        #                 for tr in triangles])
        
        if len(triangles) >= mintrials and max((len(v) for v,f,d in triangles)) >= minpoints:
            break

        if i >= maxiter:
            break

    # of all the trial triangles, choose only the one with lowest diff and longest chain of points
    triangles = sorted(triangles, key=lambda(v,f,d): (d,-len(v)) )
    orignames,origcoords = [],[]
    matchnames,matchcoords = [],[]
    print '\n'+'Final matchset:'
    for tri,f,diff in triangles[:1]: # only the first best triangle is used
        for (n,c),(mn,mc) in zip(tri, zip(f['properties']['combination'], f['geometry']['coordinates'][0])):
            print 'final',n,c,mn,mc
            if c in origcoords or mc in matchcoords: continue
            orignames.append(n)
            origcoords.append(c)
            matchnames.append(mn)
            matchcoords.append(mc)

    print 'final diff', diff
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)



        
