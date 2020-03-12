
from . import segmentation
from . import textdetect
from . import textgroup
from . import toponyms

from . import shapematch
from . import triangulate

from . import transforms
from . import accuracy
from . import imwarp

import pythongis as pg

import PIL, PIL.Image

import datetime
import time
import os
import json
import itertools


### FUNCS FOR DIFFERENT STAGES

def image_partitioning(im):
    ################
    # Image partitioning
    
    # partition image
    mapp_poly,box_polys = segmentation.image_segments(im)

    # create as feature collection (move to image_segments()?)
    seginfo = {'type': 'FeatureCollection',
               'features': []}
    
    # (map)
    if mapp_poly is not None:
        mapp_geoj = {'type': 'Polygon',
                     'coordinates': [ [tuple(p[0].tolist()) for p in mapp_poly] ]}
        props = {'type':'Map'}
        feat = {'type': 'Feature', 'properties': props, 'geometry': mapp_geoj}
        seginfo['features'].append(feat)
    
    # (boxes)
    if box_polys:
        boxes_geoj = [{'type': 'Polygon',
                     'coordinates': [ [tuple(p[0].tolist()) for p in box] ]}
                      for box in box_polys]
        for box_geoj in boxes_geoj:
            props = {'type':'Box'}
            feat = {'type': 'Feature', 'properties': props, 'geometry': box_geoj}
            seginfo['features'].append(feat)

    return seginfo

def text_detection(text_im, textcolor, colorthresh, textconf, sample, seginfo):
    ###############
    # Text detection
    
    # detect text
    print '(detecting text)'
    if textcolor and not isinstance(textcolor, list):
        textcolor = [textcolor]
    texts = textdetect.auto_detect_text(text_im, textcolors=textcolor, colorthresh=colorthresh, textconf=textconf, sample=sample, seginfo=seginfo)
    toponym_colors = set((r['color'] for r in texts))

    # deduplicate overlapping texts from different colors
    # very brute force...
    if len(toponym_colors) > 1:
        print '(deduplicating texts of different colors)'
        print 'textlen',len(texts)
        # for every combination of text colors
        for col,col2 in itertools.combinations(toponym_colors, 2):
            coltexts = [r for r in texts if r['color'] == col]
            coltexts2 = [r for r in texts if r['color'] == col2]
            print 'comparing textcolor',map(int,col),len(coltexts),'with',map(int,col2),len(coltexts2)
            # we got two different colored groups of text
            for r in coltexts:
                for r2 in coltexts2:
                    # find texts that overlap
                    if not (r['left'] > (r2['left']+r2['width']) \
                            or (r['left']+r['width']) < r2['left'] \
                            or r['top'] > (r2['top']+r2['height']) \
                            or (r['top']+r['height']) < r2['top'] \
                            ):
                        # drop the one with the poorest color match
                        #print 'found duplicate texts of different colors, dropping worst match'
                        #print r
                        #print r2
                        #text_im.crop((r['left'], r['top'], r['left']+r['width'], r['top']+r['height'])).show()
                        #text_im.crop((r2['left'], r2['top'], r2['left']+r2['width'], r2['top']+r2['height'])).show()
                        if r2['color_match'] > r['color_match']:
                            r2['drop'] = True
                        else:
                            r['drop'] = True
        texts = [r for r in texts if not r.get('drop')]
        print 'textlen deduplicated',len(texts)

    # connect texts
    print '(connecting texts)'
    grouped = []
    # connect each color texts separately
    for col in toponym_colors:
        coltexts = [r for r in texts if r['color'] == col]
        # divide into lower and upper case subgroups
        # upper = more than half of alpha characters is uppercase (to allow for minor ocr upper/lower errors)
        lowers = []
        uppers = []
        for text in coltexts:
            alphachars = text['text_alphas']
            isupper = len([ch for ch in alphachars if ch.isupper()]) > (len(alphachars) / 2.0) 
            if isupper:
                uppers.append(text)
            else:
                lowers.append(text)
        # connect lower and upper case texts separately
        if len(lowers) > 1:
            grouped.extend( textgroup.connect_text(lowers) )
        if len(uppers) > 1:
            grouped.extend( textgroup.connect_text(uppers) )
    texts = grouped

    # store metadata
    textinfo = {'type': 'FeatureCollection', 'features': []}
    for r in texts:
        x1,y1,x2,y2 = r['left'], r['top'], r['left']+r['width'], r['top']+r['height']
        box = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
        geoj = {'type':'Polygon', 'coordinates':[box]}
        props = dict(r)
        feat = {'geometry':geoj, 'properties':props}
        textinfo['features'].append(feat)

    return textinfo

def toponym_selection(im, textinfo, seginfo):
    ################
    # Toponym selection
    texts = [f['properties'] for f in textinfo['features']]

    # filter toponym candidates
    print 'filtering toponym candidates'
    topotexts = toponyms.filter_toponym_candidates(texts, seginfo)

    # text anchor points
    print 'determening toponym anchors'
    toponym_colors = set((r['color'] for r in topotexts))
    anchored = []
    # threshold img to black pixels only
    # (anchors are usually thick and almost always black, so not as affected by color blending as text)
    diff = segmentation.color_difference(segmentation.quantize(im), (0,0,0))
    diff[diff > 25] = 255
    anchor_im = PIL.Image.fromarray(diff)
    # detect anchors
    texts = toponyms.detect_toponym_anchors(anchor_im, topotexts)

    # create control points from toponyms
    points = [(r['text_clean'], r['anchor']) for r in texts if 'anchor' in r] # if r['function']=='placename']

    # store metadata
    toponyminfo = {'type': 'FeatureCollection', 'features': []}
    for name,p in points:
        geoj = {'type':'Point', 'coordinates':p}
        props = {'name':name}
        feat = {'geometry':geoj, 'properties':props}
        toponyminfo['features'].append(feat)

    return toponyminfo

def match_control_points(toponyminfo, matchthresh, db, source, **kwargs):
    ###############
    # Control point matching
    points = [(f['properties']['name'],f['geometry']['coordinates']) for f in toponyminfo['features']]

    # find matches
    origs,matches = triangulate.find_matches(points, matchthresh, db=db, source=source, **kwargs)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)

    # store metadata
    gcps_matched_info = {'type': 'FeatureCollection', 'features': []}
    for (oname,ocoord),(mname,mcoord) in zip(origs,matches):
        geoj = {'type':'Point', 'coordinates':mcoord}
        props = {'origname':oname, 'origx':ocoord[0], 'origy':ocoord[1],
                 'matchname':mname, 'matchx':mcoord[0], 'matchy':mcoord[1]}
        feat = {'geometry':geoj, 'properties':props}
        gcps_matched_info['features'].append(feat)

    return gcps_matched_info

def estimate_transform(gcps_matched_info, warp_order, residual_type):
    #################
    # Transform Estimation

    orignames = [f['properties']['origname'] for f in gcps_matched_info['features']]
    matchnames = [f['properties']['matchname'] for f in gcps_matched_info['features']]
    origcoords = [(f['properties']['origx'],f['properties']['origy']) for f in gcps_matched_info['features']]
    matchcoords = [(f['properties']['matchx'],f['properties']['matchy']) for f in gcps_matched_info['features']]
    tiepoints = zip(origcoords, matchcoords)

    if warp_order:
        # setup
        trans = transforms.Polynomial(order=warp_order)
        if residual_type == 'geographic':
            invert = False
            distance = 'geodesic'
        elif residual_type == 'pixels':
            invert = True
            distance = 'eucledian'
        else:
            raise ValueError
        pixels,coords = zip(*tiepoints)

        # initial rmse
        err,resids = accuracy.model_accuracy(trans, pixels, coords,
                                             leave_one_out=True,
                                             invert=invert, distance=distance,
                                             accuracy='rmse')
        print '{} points, RMSE: {}'.format(len(pixels), err)

        # enforce some minimum residual? 
        # ... 

        # auto drop points that best improve model
        print 'dropping points to improve model'
        trans, pixels, coords, err, resids = accuracy.auto_drop_models(trans, pixels, coords,
                                                                     improvement_ratio=0.10,
                                                                     minpoints=None,
                                                                     leave_one_out=True,
                                                                     invert=invert, distance=distance,
                                                                     accuracy='rmse')
        tiepoints = zip(pixels, coords)
        print '{} points, RMSE: {}'.format(len(pixels), err)

    else:
        # setup
        trytrans = [transforms.Polynomial(order=1), transforms.Polynomial(order=2), transforms.Polynomial(order=3)]
        if residual_type == 'geographic':
            invert = False
            distance = 'geodesic'
        elif residual_type == 'pixels':
            invert = True
            distance = 'eucledian'
        else:
            raise ValueError
        pixels,coords = zip(*tiepoints)

        # initial points
        print '{} points'.format(len(pixels))

        # auto get optimal transform
        print 'autodetecting optimal transform'
        # TODO: maybe allow improvement_ratio and minpoints params
        trans, pixels, coords, err, resids = accuracy.auto_choose_model(pixels, coords, trytrans, invert=invert, distance=distance, accuracy='rmse')
        tiepoints = zip(pixels, coords)
        print '{} points, RMSE: {}'.format(len(pixels), err)

    # estimate final forward and backward transforms for image warping
    (cols,rows),(xs,ys) = zip(*pixels),zip(*coords)
    forward = trans.copy()
    forward.fit(cols,rows,xs,ys)
    backward = trans.copy()
    backward.fit(cols,rows,xs,ys, invert=True)

    # store metadata

    # first gcps
    gcps_final_info = {'type': 'FeatureCollection', 'features': []}
    for ocoord,mcoord,res in zip(pixels,coords,resids):
        i = list(zip(origcoords, matchcoords)).index((ocoord,mcoord))
        oname = orignames[i]
        mname = matchnames[i]
        geoj = {'type':'Point', 'coordinates':mcoord}
        props = {'origname':oname, 'origx':ocoord[0], 'origy':ocoord[1],
                 'matchname':mname, 'matchx':mcoord[0], 'matchy':mcoord[1],
                 'residual':res, 'residual_type':residual_type}
        feat = {'geometry':geoj, 'properties':props}
        gcps_final_info['features'].append(feat)

    # then transforms
    # NOTE: metadata reports only RMSE error type with leave_one_out=True, maybe allow user customizing this? 
    if invert:
        # get forward error, backward is already calculated
        berr,bresids = err,resids 
        ferr,fresids = accuracy.model_accuracy(trans, pixels, coords,
                                             leave_one_out=True,
                                             invert=False, distance='geodesic',
                                             accuracy='rmse')
    else:
        # get backward error, forward is already calculated
        ferr,fresids = err,resids 
        berr,bresids = accuracy.model_accuracy(trans, pixels, coords,
                                             leave_one_out=True,
                                             invert=True, distance='eucledian',
                                             accuracy='rmse')
    
    forward_info = {'model': forward.info(),
                    'error': ferr,
                    'residuals': list(fresids),
                    # hardcoded
                    'error_type': 'rmse',
                    'leave_one_out': True}
    backward_info = {'model': backward.info(),
                     'error': berr,
                     'residuals': list(bresids),
                     # hardcoded
                     'error_type': 'rmse', 
                     'leave_one_out': True}
    transinfo = {'forward': forward_info,
                 'backward': backward_info}

    return transinfo,gcps_final_info

def warp(mapp_im, transinfo):
    #################
    # Warping

    # load the transforms
    forward = transforms.from_json(transinfo['forward']['model'])
    backward = transforms.from_json(transinfo['backward']['model'])
    
    # warp the image
    wim,aff = imwarp.warp(mapp_im, forward, backward) # warp

    # store metadata
    warp_info = {'image':wim,
                 'affine':aff}

    return warp_info



### MAIN FUNC

def automap(inpath, outpath=True, matchthresh=0.1, textcolor=None, colorthresh=25, textconf=60, sample=False, db=None, source='gns', warp_order=None, residual_type='pixels', max_residual=None, debug=False, **kwargs):
    info = dict()
    start = time.time()

    timinginfo = dict()
    timinginfo['start'] = datetime.datetime.now().isoformat()
    
    params = dict(inpath=inpath,
                  outpath=outpath,
                  matchthresh=matchthresh,
                  textcolor=textcolor,
                  colorthresh=colorthresh,
                  textconf=textconf,
                  sample=sample,
                  source=source,
                  warp_order=warp_order,
                  residual_type=residual_type,
                  )
    info['params'] = params


    
    # load image
    print 'loading image', inpath
    im = PIL.Image.open(inpath).convert('RGB')



    # determine various paths
    # outpath can be string, True, or False/None
    infold,infil = os.path.split(inpath)
    infil,ext = os.path.splitext(infil)
    if outpath:
        # output
        if outpath is True:
            # auto, relative to inpath
            outfold = infold
            outfil = infil + '_georeferenced'
        else:
            # relative to manual outpath
            outfold,outfil = os.path.split(outpath)
    else:
        # dont output, but still need for debug
        # relative to inpath
        outfold = infold
        outfil = infil + '_georeferenced'

    outfil,ext = os.path.splitext(outfil)

    


    ################
    # Image partitioning
    print '\n' + 'image segmentation'

    # remove unwanted parts of image? 
    text_im = im
##    if mapp_poly is not None:
##        text_im = segmentation.mask_image(text_im, mapp_poly)
##    for box in box_polys:
##        text_im = segmentation.mask_image(text_im, box, invert=True)
##    #text_im.show()
    
    # partition image
    t = time.time()
    seginfo = image_partitioning(text_im)

    # store timing
    elaps = time.time() - t
    timinginfo['segmentation'] = elaps

    # store metadata
    info['segmentation'] = seginfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)
    



    ###############
    # Text detection
    
    # detect text
    print '\n' + 'detecting text'
    t = time.time()
    textinfo = text_detection(text_im, textcolor, colorthresh, textconf, sample, seginfo)

    # store timing
    elaps = time.time() - t
    timinginfo['text_recognition'] = elaps

    # store metadata
    info['text_recognition'] = textinfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    ################
    # Toponym selection

    # text anchor points
    print '\n' + 'seleting toponyms with anchor points'
    t = time.time()
    toponyminfo = toponym_selection(im, textinfo, seginfo)

    # store timing
    elaps = time.time() - t
    timinginfo['toponym_candidates'] = elaps

    # store metadata
    info['toponym_candidates'] = toponyminfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)

    


    ###############
    # Control point matching

    # find matches
    print '\n' + 'finding matches'
    t = time.time()
    gcps_matched_info = match_control_points(toponyminfo, matchthresh, db, source, **kwargs)

    # store timing
    elaps = time.time() - t
    timinginfo['gcps_matched'] = elaps

    # store metadata
    info['gcps_matched'] = gcps_matched_info

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    #################
    # Transform Estimation

    # estimate the transform and return final gcps
    print '\n' + 'estimating transformation'
    t = time.time()
    transinfo,gcps_final_info = estimate_transform(gcps_matched_info, warp_order, residual_type)

    # store timing
    elaps = time.time() - t
    timinginfo['transform_estimation'] = elaps

    # store metadata
    info['gcps_final'] = gcps_final_info
    info['transform_estimation'] = transinfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    #################
    # Warping
    print '\n' + 'warping'
    print '{} points, warp_method={}'.format(len(gcps_final_info['features']), transinfo['forward']['model'])

    # mask the image before warping
    mapp_im = im
    #if mapp_poly is not None:
    #    mapp_im = segmentation.mask_image(im.convert('RGBA'), mapp_poly) # map region

    # warp the image
    t = time.time()
    warp_info = warp(mapp_im, transinfo)

    # store timing
    elaps = time.time() - t
    timinginfo['warping'] = elaps

    # store metadata
    info['warping'] = warp_info

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)



    #############
    # Finished

    total_time = time.time() - start
    timinginfo['total'] = total_time
    info['timings'] = timinginfo
    
    print '\n'+'finished!'
    print 'total runtime: {:.1f} seconds \n'.format(total_time)



    ##############
    # Save output?

    if outpath:
        print '\n' + 'saving output'
        
        # warped image
        pth = os.path.join(outfold, outfil + '.tif') # suffix already added (manual or auto, see top)
        rast = pg.RasterData(image=warp_info['image'], affine=warp_info['affine']) # to geodata
        rast.save(pth)

        # final control points
        pth = os.path.join(outfold, outfil+'_controlpoints.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['gcps_final'], writer)

        # transformation
        pth = os.path.join(outfold, outfil+'_transform.json')
        with open(pth, 'w') as writer:
            json.dump(info['transform_estimation'], writer)

    if debug:
        print '\n' + 'saving debug data'

        # segmentation
        pth = os.path.join(outfold, outfil+'_debug_segmentation.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['segmentation'], writer)

        # text recognition
        pth = os.path.join(outfold, outfil+'_debug_text.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['text_recognition'], writer)

        # toponym candidates
        pth = os.path.join(outfold, outfil+'_debug_text_toponyms.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['toponym_candidates'], writer)

        # gcps matched
        pth = os.path.join(outfold, outfil+'_debug_gcps_matched.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['gcps_matched'], writer)

        # timings
        pth = os.path.join(outfold, outfil+'_debug_timings.json')
        with open(pth, 'w') as writer:
            json.dump(info['timings'], writer)



    ##############
    # Return results metadata

    return info




    
