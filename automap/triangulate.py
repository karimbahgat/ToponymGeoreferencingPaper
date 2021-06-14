"""
"""

import itertools

from . import patternmatch
from . import geocode
from . import transforms
from . import accuracy


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

def find_matchsets(test, thresh=0.25, minpoints=8, mintrials=8, maxiter=10000, maxcandidates=None, n_combi=3, db=None, source='best', debug=False):
    # filter to those that can be geocoded
    print('geocode and filter')
    coder = geocode.OptimizedCoder(db)
    
    testres = []
    for nxtname,nxtpos in test:
        print('geocoding',nxtname)
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
            print('EXCEPTION:', err)
        
    #testres = [(nxtname,nxtpos,res)
    #           for nxtname,nxtpos,res in testres if res and len(res)<10]
    #testres = [(nxtname,nxtpos,res[:maxcandidates])
    #           for nxtname,nxtpos,res in testres]

    # print names to be tested
    for nxtname,nxtpos,res in testres:
        print(nxtname,len(res))

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

    print('\n'+'finding all possible triangles of {} possible combinations'.format(len(combis)))
    resultsets = []
    for i,tri in enumerate(combis):
        if debug:
            print('-----')
            print('try triangle %s of %s' % (i, len(combis)))
            print('\n'.join([repr((tr[0],len(tr[2]))) for tr in tri]))
        # try triang
        best = None
        names,positions,candidates = zip(*tri)
        try: best = triangulate(coder,
                                names,
                                positions,
                                candidates,
                                flipy=True)
        except Exception as err: print('EXCEPTION RAISED:',err)
        if best:
            f,diff,diffs = best[0]
            #print f
            if debug:
                print('error:', round(diff,6))
            if diff < thresh:
                if debug:
                    print('TRIANGLE FOUND')
                resultset = [tr[:2] for tr in tri]

                # incrementally add remaining points
                for nxtname,nxtpos,candidates in testres:
                    if debug:
                        print('trying to add incrementally:',nxtname,nxtpos)
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
                                           list(zip(orignames,origcoordsflip)),
                                           list(zip(matchnames,matchcoords)),
                                           (nxtname,nxtposflip),
                                           candidates)
                    if not best:
                        # ...? 
                        continue
                    mf,mdiff,mdiffs = best[0]
                    if mdiff < thresh:
                        if debug:
                            print('ADDING')
                        resultset.append((nxtname,nxtpos))
                        f = mf
                
                print('\n'+'MATCHES FOUND (point pattern error=%r)' % round(diff,6))
                print('>>>', repr([n for n,p in resultset]),'-->',[n[:15] for n in f['properties']['combination']])

                resultsets.append((resultset,f,diff))
                
        if debug:
            print('%s resultsets so far:' % len(resultsets))
        
        #print '\n>>>'.join([repr((round(tr[2],6),[n for n,p in tr[0]],'-->',[n[:15] for n in tr[1]['properties']['combination']]))
        #                 for tr in triangles])
        
        if len(resultsets) >= mintrials and max((len(r) for r,f,d in resultsets)) >= minpoints:
            break

        if i >= maxiter:
            break

    return resultsets

def best_matchset(matchsets):
    resultsets = matchsets # rename for internal use

    ######
    # BASIC COMPARISONS
    
    # hmmm, collect all corresponding point pairs and match stats across sets with extra info? 
    # (this is just used for easier and more stable lookup later)
    # first list all orig-match point pairs
##    pointpairs = []
##    for origpointset,matchpointset,setdiff in zip(origpointsets, matchpointsets, diffs):
##        for origpoint,matchpoint in zip(origpointset, matchpointset):
##            pointpairs.append((origpoint,matchpoint,origpointset,matchpointset,setdiff))
##    # then create dict of match stats for each matchpoint
##    matchpoint_stats = {}
##    groupby = lambda x: x[1]
##    for matchpoint,group in itertools.groupby(sorted(pointpairs, key=groupby), key=groupby):
##        group = list(group)
##        setlength,setdiff = group[-2:]
##        matchfreq = len(group)
##        matchpoint_stats[matchpoint] = (setlength,setdiff,matchfreq)
##    # collect some global trial stats
##    allorigpoints,allmatchpoints,allsetlengths,allsetdiffs = zip(*pointpairs)
##    avgsetdiff = sum(allsetdiffs)/float(len(allsetdiffs))
##    avgsetlength = sum(allsetlengths)/float(len(allsetlengths))
##    medsetlength = min(allsetlengths) + (max(allsetlengths)-min(allsetlengths))/2.0 #sorted(allsetlengths)[len(allsetlengths)//2]
##    print 'avgsetdiff:',avgsetdiff,'avgsetlength:',avgsetlength,'medsetlength:',medsetlength

    # produce final (choose best match across all trials)
    # v1: look at all, choose most frequent for each
##    print '\n'+'Final matchset (across trials):'
##    orignames,origcoords = [],[]
##    matchnames,matchcoords = [],[]
##    origpointsets,matchpointsets,diffs = zip(*resultsets)
##    matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##    # loop each input point
##    for n,c,_ in testres:
##        # find all matches
##        matches = []
##        for origpointset,matchpointset in zip(origpointsets, matchpointsets):
##            for origpoint,matchpoint in zip(origpointset, matchpointset):
##                if origpoint == (n,c):
##                    matches.append(matchpoint)
##        # choose most frequent match
##        matchpointcounts = [(matchpoint,len(list(group)))
##                            for matchpoint,group in itertools.groupby(sorted(matches))]
##        for (mn,mc),count in sorted(matchpointcounts, key=lambda x: -x[1]):
##            break 
##        # final control point
##        if 1:#(count / float(len(resultsets))) >= 0.5:
##            # only points that show up in over half of all trials
##            orignames.append(n)
##            origcoords.append(c)
##            matchnames.append(mn)
##            matchcoords.append(mc)
##            print 'final',n,c,mn[:100],mc,'(count {})'.format(count)

    # produce final (choose best match across all trials)
    # v2: choose ones belonging to longest set (above say avg) and lowest diff
##    print '\n'+'Final matchset (across trials):'
##    orignames,origcoords = [],[]
##    matchnames,matchcoords = [],[]
##    origpointsets,matchpointsets,diffs = zip(*resultsets)
##    matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##    # collect all corresponding point pairs across sets with extra info
##    pointpairs = []
##    for origpointset,matchpointset,setdiff in zip(origpointsets, matchpointsets, diffs):
##        for origpoint,matchpoint in zip(origpointset, matchpointset):
##            setlength = len(origpointset)
##            pointpairs.append((origpoint,matchpoint,setlength,setdiff))
##    # collect some stats
##    allorigpoints,allmatchpoints,allsetlengths,allsetdiffs = zip(*pointpairs)
##    avgsetdiff = sum(allsetdiffs)/float(len(allsetdiffs))
##    avgsetlength = sum(allsetlengths)/float(len(allsetlengths))
##    medsetlength = min(allsetlengths) + (max(allsetlengths)-min(allsetlengths))/2.0 #sorted(allsetlengths)[len(allsetlengths)//2]
##    print 'avgsetdiff:',avgsetdiff,'avgsetlength:',avgsetlength,'medsetlength:',medsetlength
##    # count match frequency for each matchpoint
##    matchpoint_counts = {}
##    groupby = lambda x: x[1]
##    for matchpoint,group in itertools.groupby(sorted(pointpairs, key=groupby), key=groupby):
##        matchpoint_counts[matchpoint] = len(list(group))
##    # group all possible matchpoints with each origpoint
##    groupby = lambda x: x[0] # origpoint
##    for origpoint,group in itertools.groupby(sorted(pointpairs, key=groupby), key=groupby):
##        print '>>>', origpoint[0], origpoint[1]
##        group = list(group)
##        # add in match frequency
##        group = [(origpoint,matchpoint,setlength,matchpoint_counts[matchpoint],setdiff)
##                 for origpoint,matchpoint,setlength,setdiff in group]
##        # sort by setlength then matchfreq then setdiff
##        sortby = lambda x: (-x[-3], -x[-2], x[-1])
##        for i,(origpoint,matchpoint,setlength,matchfreq,setdiff) in enumerate(sorted(group, key=sortby)):
##            n,c = origpoint
##            mn,mc = matchpoint
##            included = False
##            if i == 0:
##                # condition
##                if setlength >= medsetlength: #and avgsetdiff >= avgsetdiff:
##                    # add to result
##                    included = True
##                    orignames.append(n)
##                    origcoords.append(c)
##                    matchnames.append(mn)
##                    matchcoords.append(mc)
##            if included:
##                print '  ',included,mn[:50],mc,'(setlength {}, matchfreq {}, setdiff {})'.format(setlength, matchfreq, setdiff)

    # additional checks:
    #add only if setlength above avg?
    #try adding to set and keep only if doesn't worsen set diff too much
    #find central point and drop stdev distance outliers (away from central cluster)
    # alteernatively:
    #maybe take the best match as the starting point and incrementally add from all possible matches?
    # then later on:
    #run LOO autodrop until the end and choose the global optimal (but will this lead to small sample)
    
    # produce final (longest and lowest diff) OLD
##    print '\n'+'Final matchset (longest and lowest diff):'
##    orignames,origcoords = [],[]
##    matchnames,matchcoords = [],[]
##    resultsets = sorted(resultsets, key=lambda(v,f,d): (-len(v),d) )
##    bestset,bestf,bestdiff = resultsets[0] # only the first best triangle is used
##    for (n,c),(mn,mc) in zip(bestset, zip(bestf['properties']['combination'], bestf['geometry']['coordinates']) ):
##        print 'final',n,c,mn[:50],mc
##        if c in origcoords or mc in matchcoords:
##            # in case of duplicates? 
##            continue
##        orignames.append(n)
##        origcoords.append(c)
##        matchnames.append(mn)
##        matchcoords.append(mc)
##    print 'final length', len(bestset), 'diff', diff

    # produce final (longest and lowest diff) NEW
##    print '\n'+'Final matchset (longest and lowest diff):'
##    orignames,origcoords = [],[]
##    matchnames,matchcoords = [],[]
##    origpointsets,matchpointsets,diffs = zip(*resultsets)
##    matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##    # get final set
##    sortby = lambda(origpointset,matchpointset,setdiff): (-len(origpointset),setdiff)
##    best_origpointset,best_matchpointset,best_setdiff = sorted(zip(origpointsets,matchpointsets,diffs), key=sortby)[0]
##    for (n,c),(mn,mc) in zip(best_origpointset,best_matchpointset):
##        print 'final',n,c,mn[:50],mc
##        if (n,c) in zip(orignames,origcoords) or (mn,mc) in zip(matchnames,matchcoords):
##            # in case of duplicates? 
##            continue
##        orignames.append(n)
##        origcoords.append(c)
##        matchnames.append(mn)
##        matchcoords.append(mc)
##    print 'final length', len(best_origpointset), 'diff', best_setdiff




    ######
    # FULL MODEL EST COMPARISONS

    # prep lists
    print('\n'+'Comparing matchsets (full model comparison):')
    origpointsets,matchpointsets,diffs = zip(*resultsets)
    matchpointsets = [list(zip(f['properties']['combination'], f['geometry']['coordinates'])) for f in matchpointsets]

    # for each set, estimate the optimal polynomial model
    trytrans = [transforms.Polynomial(order=1), transforms.Polynomial(order=2), transforms.Polynomial(order=3)]
    results = []
    for i,(origpointset,matchpointset) in enumerate(zip(origpointsets,matchpointsets)):
        origpointnames,origpointcoords = zip(*origpointset)
        matchpointnames,matchpointcoords = zip(*matchpointset)
        res = accuracy.auto_choose_model(origpointcoords, matchpointcoords, trytrans, refine_outliers=False)
        print('matchset', i, 'length', len(origpointset), 'model', res[0], 'error', res[-2])
        results.append((i,res))

    # get the set with the lowest model error
    print('\n'+'Final matchset (full model comparison):')
    #sortby = lambda(i,res): (-len(res[1]),res[-2]) # sort by setlength and then model error
    sortby = lambda i_res: i_res[1][-2] # sort by model error only
    best = sorted(results, key=sortby)[0] 
    best_i,(trans, inpoints, outpoints, err, resids) = best
    print('chosen matchset num', best_i, 'model type', trans, 'model error', err)
    orignames,origcoords = zip(*origpointsets[best_i])
    matchnames,matchcoords = zip(*matchpointsets[best_i])
    for n,c,mn,mc in zip(orignames,origcoords,matchnames,matchcoords):
        print('final',n,c,mn[:50],mc)




    ######
    # DEBUG

    # debug trial matchsets by visualizing on map
##    import traceback
##    try:
##        import pythongis as pg
##        m = pg.renderer.Map()
##        m.add_layer(r"C:\Users\kimok\Desktop\BIGDATA\priocountries\priocountries.shp")
##        for origpointset,matchpointset,diff in resultsets:
##            f = matchpointset
##            matchpointset = zip(f['properties']['combination'], f['geometry']['coordinates'])
##            d = pg.VectorData(fields=['matchname','setdiff'])
##            for mn,mc in matchpointset:
##                d.add_feature([mn,diff], {'type':'Point', 'coordinates':mc})
##            m.add_layer(d, fillsize={'key':'setdiff','breaks':[0,0.05,0.1,0.26]}, fillopacity=0.5)
##        m.view()
##    except:
##        traceback.print_exc()

    # debug trial match counts OLD
##    print '\n'+'Debug matchset counts across trials'
##    import traceback
##    try:
##        final_origpointset,final_matchpointset,final_setdiff = best_origpointset,best_matchpointset,best_setdiff # varnames may need to be changed for the other methods
##        origpointsets,matchpointsets,diffs = zip(*resultsets)
##        matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##        for origname,origpos,_ in testres:
##            print '>>>', origname, origpos
##            matches = []
##            for origpointset,matchpointset in zip(origpointsets, matchpointsets):
##                for origpoint,matchpoint in zip(origpointset, matchpointset):
##                    if origpoint == (origname,origpos):
##                        matches.append(matchpoint)
##            matchpointcounts = [(matchpoint,len(list(group)))
##                                for matchpoint,group in itertools.groupby(sorted(matches))]
##            for i,((matchname,matchcoord),count) in enumerate(sorted(matchpointcounts, key=lambda x: -x[1])):
##                included = (matchname,matchcoord) in final_matchpointset
##                print '  ', included, 'count', count, matchname[:100], matchcoord, 'diff', diffs[i], 'setlength', len(matchpointsets[i])
##    except:
##        traceback.print_exc()

    # debug trial match counts NEW
##    print '\n'+'Debug matchset counts across trials'
##    import traceback
##    try:
##        origpointsets,matchpointsets,diffs = zip(*resultsets)
##        matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##        # collect all corresponding point pairs across sets with extra info
##        pointpairs = []
##        for origpointset,matchpointset,setdiff in zip(origpointsets, matchpointsets, diffs):
##            for origpoint,matchpoint in zip(origpointset, matchpointset):
##                setlength = len(origpointset)
##                pointpairs.append((origpoint,matchpoint,setlength,setdiff))
##        # collect some stats
##        allorigpoints,allmatchpoints,allsetlengths,allsetdiffs = zip(*pointpairs)
##        avgsetdiff = sum(allsetdiffs)/float(len(allsetdiffs))
##        avgsetlength = sum(allsetlengths)/float(len(allsetlengths))
##        medsetlength = min(allsetlengths) + (max(allsetlengths)-min(allsetlengths))/2.0 #sorted(allsetlengths)[len(allsetlengths)//2]
##        print 'avgsetdiff:',avgsetdiff,'avgsetlength:',avgsetlength,'medsetlength:',medsetlength
##        # count match frequency for each matchpoint
##        matchpoint_counts = {}
##        groupby = lambda x: x[1]
##        for matchpoint,group in itertools.groupby(sorted(pointpairs, key=groupby), key=groupby):
##            matchpoint_counts[matchpoint] = len(list(group))
##        # group all possible matchpoints with each origpoint
##        groupby = lambda x: x[0] # origpoint
##        for origpoint,group in itertools.groupby(sorted(pointpairs, key=groupby), key=groupby):
##            print '>>>', origpoint[0], origpoint[1]
##            group = list(group)
##            # add in match frequency
##            group = [(origpoint,matchpoint,setlength,matchpoint_counts[matchpoint],setdiff)
##                     for origpoint,matchpoint,setlength,setdiff in group]
##            # sort by setlength then setdiff
##            sortby = lambda x: (-x[-3], x[-1])
##            for i,(origpoint,matchpoint,setlength,matchfreq,setdiff) in enumerate(sorted(group, key=sortby)):
##                n,c = origpoint
##                mn,mc = matchpoint
##                included = matchpoint in zip(matchnames,matchcoords)
##                print '  ',included,mn[:50],mc,'(setlength {}, matchfreq {}, setdiff {})'.format(setlength, matchfreq, setdiff)
##    except:
##        traceback.print_exc()

    # debug trial match counts NEW 2 (NOT FINISHED)
##    print '\n'+'Debug matchset counts across trials'
##    import traceback
##    try:
##        final_origpointset,final_matchpointset,final_setdiff = best_origpointset,best_matchpointset,best_setdiff # varnames may need to be changed for the other methods
##        origpointsets,matchpointsets,diffs = zip(*resultsets)
##        matchpointsets = [zip(f['properties']['combination'], f['geometry']['coordinates']) for f in matchpointsets]
##        for origname,origpos,_ in testres:
##            print '>>>', origname, origpos
##            info = []
##            for origpointset,matchpointset,setdiff in zip(origpointsets, matchpointsets, diffs):
##                for origpoint,matchpoint in zip(origpointset, matchpointset):
##                    if origpoint == (origname,origpos):
##                        matchfreq = len([1 for mset in matchpointsets if matchpoint in mset])
##                        info.append((matchpoint,origpointset,matchpointset,setdiff))
##            groupedinfo = 
##            sortby = lambda(matchpoint,origpointset,matchpointset,setdiff): (-len(matchpointset),setdiff)
##            for matchpoint,origpointset,matchpointset,setdiff in sorted(info, key=sortby):
##                count = '???'
##                setlength = len(matchpointset)
##                included = matchpoint in final_matchpointset
##                matchname,matchcoord = matchpoint
##                print '  ', included, matchname[:100], matchcoord, 'setlength', setlength, 'count', count, 'diff', setdiff
##    except:
##        traceback.print_exc()

    return list(zip(orignames, origcoords)), list(zip(matchnames, matchcoords))



        
