"""
"""

import itertools

from . import shapematch
from . import patternmatch
from . import geocode


def triangulate(coder, names, positions, matchcandidates=None, flipy=False):
    assert len(names) == len(positions)
    assert len(names) >= 3

    # reverse ys due to flipped image coordsys
    if flipy:
        maxy = max((y for x,y in positions))
        positions = [(x,maxy-y) for x,y in positions]

    # define input names/positions as a polygon feature
    findpattern = {'type': 'Feature',
                   'properties': {'combination': names,
                                  },
                   'geometry': {'type': 'MultiPoint',
                                'coordinates': positions
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
    combipatterns = []
    for combi in combis:
        #print '--->', combi
        # make into polygon feature
        combinames = [f['properties']['name'] for f in combi]
        combipositions = [f['geometry']['coordinates'] for f in combi]
        combipattern = {'type': 'Feature',
                       'properties': {'combination': combinames,
                                      },
                       'geometry': {'type': 'MultiPoint',
                                    'coordinates': combipositions
                                    },
                       }
        combipatterns.append(combipattern)

    # prep/normalize combipolys
    #print 'prepping'
    combipatterns = patternmatch.prep_pool(combipatterns)

    # find closest match
    #print 'finding'
    matches = patternmatch.find_best_matches(findpattern, combipatterns)
    return matches

def triangulate_add(coder, origs, matches, add, addcandidates=None):
    orignames, origpositions = zip(*origs)
    matchnames, matchpositions = zip(*matches)

    addname,addpos = add

    # define input names/positions as a multipoint feature
    findpattern = {'type': 'Feature',
                   'properties': {'combination': list(orignames) + [addname],
                                  },
                   'geometry': {'type': 'MultiPoint',
                                'coordinates': list(origpositions) + [addpos]
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
    combipatterns = []
    for m in match:
        #print '--->', combi
        # make into polygon feature
        combinames = list(matchnames) + [m['properties']['name']]
        combipositions = list(matchpositions) + [m['geometry']['coordinates']]
        combipattern = {'type': 'Feature',
                       'properties': {'combination': combinames,
                                      },
                       'geometry': {'type': 'MultiPoint',
                                    'coordinates': combipositions
                                    },
                       }
        combipatterns.append(combipattern)

    # prep/normalize combipatterns
    #print 'prepping'
    combipatterns = patternmatch.prep_pool(combipatterns)

    # find closest match
    #print 'finding'
    matches = patternmatch.find_best_matches(findpattern, combipatterns)
    return matches

def find_matches(test, thresh=0.25, minpoints=8, mintrials=8, maxiter=500, maxcandidates=None, n_combi=3, db=None, source='best', debug=False):
    # filter to those that can be geocoded
    print 'geocode and filter'
    coder = geocode.OptimizedCoder(db)

    if source == 'best' or source == 'avg':
        mintrials = 30
        maxiter = 10000
    
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
    #from random import uniform
    #combis = sorted(combis, key=lambda x: uniform(0,1))
    # sort by length of possible geocodings, ie try most unique first --> faster+accurate
    combis = sorted(combis, key=lambda gr: sum((len(res) for nxtname,nxtpos,res in gr)))

    print '\n'+'finding all possible triangles of {} possible combinations'.format(len(combis))
    resultsets = []
    for i,tri in enumerate(combis):
        if debug:
            print '-----'
            print 'try triangle %s of %s' % (i, len(combis))
            print '\n'.join([repr((tr[0],len(tr[2]))) for tr in tri])
        # try triang
        best = None
        names,positions,candidates = zip(*tri)
        try: best = triangulate(coder,
                                names,
                                positions,
                                candidates,
                                flipy=True)
        except Exception as err: print 'EXCEPTION RAISED:',err
        if best:
            f,diff,diffs = best[0]
            #print f
            if debug:
                print 'error:', round(diff,6)
            if diff < thresh:
                if debug:
                    print 'TRIANGLE FOUND'
                resultset = [tr[:2] for tr in tri]

                # incrementally add remaining points
                for nxtname,nxtpos,candidates in testres:
                    if debug:
                        print 'trying to add incrementally:',nxtname,nxtpos
                    orignames,origcoords = zip(*resultset)
                    orignames,origcoords = list(orignames),list(origcoords)
                    matchnames = list(f['properties']['combination'])
                    matchcoords = list(f['geometry']['coordinates'])
                    
                    if nxtpos in origcoords:
                        # already in the pointset
                        continue
                    maxy = max((y for x,y in origcoords))
                    maxy = max(maxy,nxtpos[1])
                    nxtposflip = (nxtpos[0],maxy-nxtpos[1])
                    origcoordsflip = [(x,maxy-y) for x,y in origcoords + [nxtpos]]
                    best = triangulate_add(coder,
                                           zip(orignames,origcoordsflip),
                                           zip(matchnames,matchcoords),
                                           (nxtname,nxtposflip),
                                           candidates)
                    if not best:
                        # ...? 
                        continue
                    mf,mdiff,mdiffs = best[0]
                    if mdiff < thresh:
                        if debug:
                            print 'ADDING'
                        resultset.append((nxtname,nxtpos))
                        f = mf
                
                print '\n'+'MATCHES FOUND (error=%r)' % round(diff,6)
                print '>>>', repr([n for n,p in resultset]),'-->',[n[:15] for n in f['properties']['combination']]

                resultsets.append((resultset,f,diff))
                
        if debug:
            print '%s resultsets so far:' % len(resultsets)
        
        #print '\n>>>'.join([repr((round(tr[2],6),[n for n,p in tr[0]],'-->',[n[:15] for n in tr[1]['properties']['combination']]))
        #                 for tr in triangles])
        
        if len(resultsets) >= mintrials and max((len(r) for r,f,d in resultsets)) >= minpoints:
            break

        if i >= maxiter:
            break

    # debug trial match counts
##    print '\n'+'Debug matchset counts across trials'
##    resultsets = sorted(resultsets, key=lambda(v,f,d): (d,-len(v)) )
##    bestset,bestf,bestdiff = resultsets[0]
##    import traceback
##    try:
##        origpointsets,matchpointsets,diffs = zip(*resultsets)
##        matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##        for origname,origpos,_ in testres:
##            print origname
##            matches = []
##            for origpointset,matchpointset in zip(origpointsets, matchpointsets):
##                for origpoint,matchpoint in zip(origpointset, matchpointset):
##                    if origpoint == (origname,origpos):
##                        matches.append(matchpoint)
##            matchpointcounts = [(matchpoint,len(list(group)))
##                                for matchpoint,group in itertools.groupby(sorted(matches))]
##            for (matchname,matchcoord),count in sorted(matchpointcounts, key=lambda x: -x[1]):
##                included = (matchname,matchcoord) in zip(bestf['properties']['combination'], bestf['geometry']['coordinates'])
##                print '  ', included, count, matchname[:100], matchcoord
##    except:
##        traceback.print_exc()

    # produce final (choose best match across all trials)
    print '\n'+'Final matchset (across trials):'
    orignames,origcoords = [],[]
    matchnames,matchcoords = [],[]
    origpointsets,matchpointsets,diffs = zip(*resultsets)
    matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
    # loop each input point
    for n,c,_ in testres:
        # find all matches
        matches = []
        for origpointset,matchpointset in zip(origpointsets, matchpointsets):
            for origpoint,matchpoint in zip(origpointset, matchpointset):
                if origpoint == (n,c):
                    matches.append(matchpoint)
        # choose most frequent match
        matchpointcounts = [(matchpoint,len(list(group)))
                            for matchpoint,group in itertools.groupby(sorted(matches))]
        for (mn,mc),count in sorted(matchpointcounts, key=lambda x: -x[1]):
            break 
        # final control point
        orignames.append(n)
        origcoords.append(c)
        matchnames.append(mn)
        matchcoords.append(mc)
        print 'final',n,c,mn[:100],mc,'(count {})'.format(count)

    # produce final (lowest diff)
##    print '\n'+'Final matchset (lowest diff):'
##    orignames,origcoords = [],[]
##    matchnames,matchcoords = [],[]
##    resultsets = sorted(resultsets, key=lambda(v,f,d): (d,-len(v)) )
##    bestset,bestf,bestdiff = resultsets[0]
##    for resultset,f,diff in resultsets[:1]: # only the first best triangle is used
##        for (n,c),(mn,mc) in zip(resultset, zip(f['properties']['combination'], f['geometry']['coordinates']) ):
##            print 'final',n,c,mn[:100],mc
##            if c in origcoords or mc in matchcoords:
##                # ...? 
##                continue
##            orignames.append(n)
##            origcoords.append(c)
##            matchnames.append(mn)
##            matchcoords.append(mc)
##    print 'final diff', diff
            
    return zip(orignames, origcoords), zip(matchnames, matchcoords)



        
