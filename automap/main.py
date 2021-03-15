
from . import segmentation
from . import textdetect
from . import textgroup
from . import toponyms

from . import triangulate

from . import transforms
from . import accuracy
from . import imwarp

import pythongis as pg

import PIL, PIL.Image
import numpy as np

import datetime
import time
import math
import os
import json
import itertools
import warnings


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

    # debug extracted segments...
##    import pythongis as pg
##    d = pg.VectorData()
##    for geoj in seginfo['features']:
##        d.add_feature([], geoj['geometry'])
##    d.view(fillcolor=None)

    return seginfo

def text_detection(text_im, textcolor, colorthresh, textconf, parallel, sample, seginfo, max_procs):
    ###############
    # Text detection
    
    # detect text
    print '(detecting text)'
    if textcolor and not isinstance(textcolor, list):
        textcolor = [textcolor]
    texts = textdetect.auto_detect_text(text_im, textcolors=textcolor, colorthresh=colorthresh, textconf=textconf, parallel=parallel, sample=sample, seginfo=seginfo, max_procs=max_procs)
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
                        #text_im.crop((r['left'], r['top'], r['left']+r['width'], r['top']+r['height'])).show()
                        #text_im.crop((r2['left'], r2['top'], r2['left']+r2['width'], r2['top']+r2['height'])).show()
                        if r2['color_match'] > r['color_match'] and not math.isnan(r2['color_match']):
                            r2['drop'] = True
                            print u'found duplicate texts of different colors, keeping "{}" (color match={:.2f}), dropping "{}" (color match={:.2f})'.format(r['text_clean'],r['color_match'],r2['text_clean'],r2['color_match'])
                        else:
                            r['drop'] = True
                            print u'found duplicate texts of different colors, keeping "{}" (color match={:.2f}), dropping "{}" (color match={:.2f})'.format(r2['text_clean'],r2['color_match'],r['text_clean'],r['color_match'])
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
        feat = {'type':'Feature', 'geometry':geoj, 'properties':props}
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
    topotexts = toponyms.detect_toponym_anchors(im, texts, topotexts)

    # create control points from toponyms
    points = [(r['text_clean'], r['anchor']) for r in topotexts if 'anchor' in r] # if r['function']=='placename']

    # store metadata
    toponyminfo = {'type': 'FeatureCollection', 'features': []}
    for name,p in points:
        geoj = {'type':'Point', 'coordinates':p}
        props = {'name':name}
        feat = {'type':'Feature', 'geometry':geoj, 'properties':props}
        toponyminfo['features'].append(feat)

    return toponyminfo

def match_control_points(toponyminfo, matchthresh, db, source, **kwargs):
    ###############
    # Control point matching
    points = [(f['properties']['name'],f['geometry']['coordinates']) for f in toponyminfo['features']]

    # find matches
    matchsets = triangulate.find_matchsets(points, matchthresh, db=db, source=source, **kwargs)
    origs,matches = triangulate.best_matchset(matchsets)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)

    # store metadata
    gcps_matched_info = {'type': 'FeatureCollection', 'features': []}
    for (oname,ocoord),(mname,mcoord) in zip(origs,matches):
        geoj = {'type':'Point', 'coordinates':mcoord}
        props = {'origname':oname, 'origx':ocoord[0], 'origy':ocoord[1],
                 'matchname':mname, 'matchx':mcoord[0], 'matchy':mcoord[1]}
        feat = {'type':'Feature', 'geometry':geoj, 'properties':props}
        gcps_matched_info['features'].append(feat)

    return gcps_matched_info

def estimate_transform(imageinfo, gcps_matched_info, warp_order, residual_type):
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

    # estimate final forward transform for image warping
    (cols,rows),(xs,ys) = zip(*pixels),zip(*coords)
    forward = trans.copy()
    forward.fit(cols,rows,xs,ys)
    
    # and backward

    # ALT1: standard inverse
    backward = trans.copy()
    backward.fit(cols,rows,xs,ys, invert=True)
    
    # ALT2: estimate slightly differently based on the bounds of the image
##    # - first get sample points at regular intervals across the image space
##    imw,imh = imageinfo['width'],imageinfo['height'] 
##    _cols = np.linspace(0, imw-1, imw) #100)
##    _rows = np.linspace(0, imh-1, imh) #100)
##    _cols,_rows = np.meshgrid(_cols, _rows)
##    _cols,_rows = _cols.flatten(), _rows.flatten()
##    # - forward predict the sample points
##    _predx,_predy = forward.predict(_cols, _rows)
##    # - then get backward transform by fitting the forward predicted sample points to the sample points
##    backward = trans.copy()
##    backward.fit(_predx, _predy, _cols, _rows)
    
    # - temporary debug: test forward vs backward resampling accuracy at regular intervals across image...
##    imw,imh = imageinfo['width'],imageinfo['height'] 
##    _cols = np.linspace(0, imw-1, imw) #100)
##    _rows = np.linspace(0, imh-1, imh) #100)
##    _cols,_rows = np.meshgrid(_cols, _rows)
##    _cols,_rows = _cols.flatten(), _rows.flatten()
##    _predx,_predy = forward.predict(_cols, _rows)
##    _cols_backpred,_rows_backpred = backward.predict(_predx, _predy)
##    print(_cols_backpred)
##    print(_rows_backpred)
##    dists = accuracy.distances(_cols, _rows, _cols_backpred, _rows_backpred)
##    print('!!! image pixel bounds', _cols.min(), _cols.max(), _rows.min(), _rows.max())
##    print('!!! image backsamp pixel bounds', _cols_backpred.min(), _cols_backpred.max(), _rows_backpred.min(), _rows_backpred.max())
##    print('!!! image backsamp max resid', dists.max())
##    print('!!! image backsamp rmse resid', accuracy.RMSE(dists))
    
    # predicted points and residuals
    xs_pred,ys_pred = forward.predict(cols, rows)
    resids_xy = accuracy.distances(xs, ys, xs_pred, ys_pred, 'geodesic')
    cols_pred,rows_pred = backward.predict(xs, ys)
    resids_colrow = accuracy.distances(cols, rows, cols_pred, rows_pred, 'eucledian')

    # store metadata

    # first gcps
    gcps_final_info = {'type': 'FeatureCollection', 'features': []}
    for col,col_pred,row,row_pred,x,x_pred,y,y_pred,res_colrow,res_xy in zip(cols,cols_pred,rows,rows_pred,xs,xs_pred,ys,ys_pred,resids_colrow,resids_xy):
        ocoord = (col,row)
        mcoord = (x,y)
        i = list(zip(origcoords, matchcoords)).index((ocoord,mcoord))
        oname = orignames[i]
        mname = matchnames[i]
        geoj = {'type':'Point', 'coordinates':mcoord}
        props = {'origname':oname, 'origx':ocoord[0], 'origy':ocoord[1], 'origx_pred':col_pred, 'origy_pred':row_pred,
                 'matchname':mname, 'matchx':mcoord[0], 'matchy':mcoord[1], 'matchx_pred':x_pred, 'matchy_pred':y_pred,
                 'origresidual':res_colrow, 'matchresidual':res_xy}
        feat = {'type':'Feature', 'geometry':geoj, 'properties':props}
        gcps_final_info['features'].append(feat)

    # then transforms
    transinfo = {'forward': forward.info(),
                 'backward': backward.info()}

    return transinfo,gcps_final_info

def calculate_errors(imageinfo, gcps_final_info):
    # first, get the model residuals
    resids_xy = [f['properties']['matchresidual'] for f in gcps_final_info['features']]
    resids_colrow = [f['properties']['origresidual'] for f in gcps_final_info['features']]
    
    # calc model errors
    errs = {}
    # first geographic
    errs['geographic'] = {}
    errs['geographic']['mae'] = accuracy.MAE(resids_xy)
    errs['geographic']['rmse'] = accuracy.RMSE(resids_xy)
    errs['geographic']['max'] = float(max(resids_xy))
    # then pixels
    errs['pixel'] = {}
    errs['pixel']['mae'] = accuracy.MAE(resids_colrow)
    errs['pixel']['rmse'] = accuracy.RMSE(resids_colrow)
    errs['pixel']['max'] = float(max(resids_colrow))
    # then percent (of image pixel dims)
    errs['percent'] = {}
    diag = math.hypot(imageinfo['width'], imageinfo['height'])
    img_radius = float(diag/2.0) # percent of half the max dist (from img center to corner)
    errs['percent']['mae'] = (errs['pixel']['mae'] / img_radius) * 100.0
    errs['percent']['rmse'] = (errs['pixel']['mae'] / img_radius) * 100.0
    errs['percent']['max'] = (errs['pixel']['max'] / img_radius) * 100.0

    # add percent residual to gcps_final_info
    for f in gcps_final_info['features']:
        pixres = f['properties']['origresidual']
        percerr = (pixres / img_radius) * 100.0
        f['properties']['percresidual'] = percerr

    return errs

def warp_image(mapp_im, transinfo):
    #################
    # Warping

    # load the transforms
    forward = transforms.from_json(transinfo['forward'])
    backward = transforms.from_json(transinfo['backward'])
    
    # warp the image
    wim,aff = imwarp.warp(mapp_im, forward, backward) # warp

    # store metadata
    warp_info = {'image':wim,
                 'affine':aff}

    return warp_info



### MAIN FUNC

def automap(im, outpath=True, matchthresh=0.25, textcolor=None, colorthresh=25, textconf=60, sample=False, parallel=False, max_procs=None, db=None, source='best', warp=True, warp_order=None, residual_type='pixels', max_residual=None, debug=False, priors=None, **kwargs):
    info = dict()
    priors = priors or dict()
    start = time.time()

    timinginfo = dict()
    timinginfo['start'] = datetime.datetime.now().isoformat()





    # determine various paths
    # outpath can be string, True, or False/None
    if isinstance(im, basestring):
        # load from path
        inpath = im
    else:
        # already a PIL image
        inpath = None
        if outpath is True:
            # deactivate auto outpath since there is no inpath to base it on
            outpath = False
        
    if outpath:
        # outputting to path
        if outpath is True:
            if inpath:
                # auto, relative to inpath
                infold,infil = os.path.split(inpath)
                infil,ext = os.path.splitext(infil)
                outfold = infold
                outfil = infil + '_georeferenced'
            else:
                # processing a preloaded image, no inpath available to determine outpath
                raise Exception('Cannot determine an appropriate outpath from a preloaded image, outpath must be set to False or specified manually.')
        else:
            # relative to manual outpath
            outfold,outfil = os.path.split(outpath)
            
        outfil,ext = os.path.splitext(outfil)
        
    else:
        # no output
        pass




    # register params
    params = dict(inpath=inpath,
                  outpath=outpath,
                  matchthresh=matchthresh,
                  textcolor=textcolor,
                  colorthresh=colorthresh,
                  textconf=textconf,
                  sample=sample,
                  parallel=parallel,
                  max_procs=max_procs,
                  source=source,
                  warp_order=warp_order,
                  residual_type=residual_type,
                  priors=priors,
                  )
    info['params'] = params

    # debug output? 
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_params.json')
        with open(pth, 'w') as writer:
            json.dump(info['params'], writer)





    # determine input image
    print '\n' + 'loading image', im
    if inpath:
        # load from path
        im = PIL.Image.open(inpath)
    else:
        # already a PIL image
        im = im
        
    if not im.mode == 'RGB':
        im = im.convert('RGB')

    imageinfo = dict(width=im.size[0],
                     height=im.size[1],
                     )
    info['image'] = imageinfo

    # debug output? 
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_image.json')
        with open(pth, 'w') as writer:
            json.dump(info['image'], writer)
            

    


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
    if 'segmentation' in priors:
        # already given
        seginfo = priors.get('segmentation')
    else:
        seginfo = image_partitioning(text_im)

    # store timing
    elaps = time.time() - t
    timinginfo['segmentation'] = elaps

    # store metadata
    info['segmentation'] = seginfo

    # debug output? 
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_segmentation.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['segmentation'], writer)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)
    



    ###############
    # Text detection
    
    # detect text
    print '\n' + 'detecting text'
    t = time.time()
    textinfo = priors.get('text_recognition', None)
    if textinfo:
        # already given
        pass
    elif priors.get('toponym_candidates', None) or priors.get('gcps_matched', None) or (priors.get('transform_estimation', None) and priors.get('gcps_final', None)):
        # later stage given, so not necessary
        pass
    else:
        textinfo = text_detection(text_im, textcolor, colorthresh, textconf, parallel, sample, seginfo, max_procs)

    # store timing
    elaps = time.time() - t
    timinginfo['text_recognition'] = elaps

    # store metadata
    info['text_recognition'] = textinfo

    # output debug?
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_text.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['text_recognition'], writer)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    ################
    # Toponym selection

    # text anchor points
    print '\n' + 'seleting toponyms with anchor points'
    t = time.time()
    toponyminfo = priors.get('toponym_candidates', None)
    if toponyminfo:
        # already given
        pass
    elif priors.get('gcps_matched', None) or (priors.get('transform_estimation', None) and priors.get('gcps_final', None)):
        # later stage given, so not necessary
        pass
    else:
        toponyminfo = toponym_selection(im, textinfo, seginfo)

    # store timing
    elaps = time.time() - t
    timinginfo['toponym_candidates'] = elaps

    # store metadata
    info['toponym_candidates'] = toponyminfo

    # output debug? 
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_text_toponyms.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['toponym_candidates'], writer)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)

    


    ###############
    # Control point matching

    # find matches
    print '\n' + 'finding matches'
    t = time.time()
    #gcps_matched_info = match_control_points(toponyminfo, matchthresh, db, source, **kwargs)
    gcps_matched_info = priors.get('gcps_matched', None)
    if gcps_matched_info:
        # already given
        pass
    elif (priors.get('transform_estimation', None) and priors.get('gcps_final', None)):
        # later stage given, so not necessary
        pass
    else:
        try:
            import traceback
            gcps_matched_info = match_control_points(toponyminfo, matchthresh, db, source, **kwargs)
        except:
            err = traceback.format_exc()
            warnings.warn('Georeferencing failed - unable to find matching control points: {}.'.format(err))
            return info

    # store timing
    elaps = time.time() - t
    timinginfo['gcps_matched'] = elaps

    # store metadata
    info['gcps_matched'] = gcps_matched_info

    # output debug? 
    if outpath and debug:
        pth = os.path.join(outfold, outfil+'_debug_gcps_matched.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['gcps_matched'], writer)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    #################
    # Transform Estimation

    # estimate the transform and return final gcps
    print '\n' + 'estimating transformation'
    t = time.time()
    transinfo = priors.get('transform_estimation', None)
    gcps_final_info = priors.get('gcps_final', None)
    if (transinfo and gcps_final_info):
        # already given
        pass
    else:
        transinfo,gcps_final_info = estimate_transform(imageinfo, gcps_matched_info, warp_order, residual_type)

    # store timing
    elaps = time.time() - t
    timinginfo['transform_estimation'] = elaps

    # store metadata
    info['gcps_final'] = gcps_final_info
    info['transform_estimation'] = transinfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)





    #################
    # Calculate model errors

    # estimate the model errors and return info
    print '\n' + 'calculating errors'
    t = time.time()
    errinfo = calculate_errors(imageinfo, gcps_final_info)

    # store timing
    elaps = time.time() - t
    timinginfo['error_calculation'] = elaps

    # store metadata
    info['error_calculation'] = errinfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)
    




    ###############
    # Bbox

    # (add in useful bbox info)
    forward = transforms.from_json(transinfo['forward'])
    x,y = forward.predict([0], [0])
    x1,y1 = float(x[0]), float(y[0])
    x,y = forward.predict([im.size[0]], [im.size[1]])
    x2,y2 = float(x[0]), float(y[0])
    info['bbox'] = [x1,y1,x2,y2]

    



    #################
    # Warping
    if warp:
        print '\n' + 'warping'
        print '{} points, warp_method={}'.format(len(gcps_final_info['features']), transinfo['forward'])

        # mask the image before warping
        mapp_im = im
        #if mapp_poly is not None:
        #    mapp_im = segmentation.mask_image(im.convert('RGBA'), mapp_poly) # map region

        # warp the image
        t = time.time()
        warp_info = warp_image(mapp_im, transinfo)

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
        if warp:
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

        # errors
        pth = os.path.join(outfold, outfil+'_errors.json')
        with open(pth, 'w') as writer:
            json.dump(info['error_calculation'], writer)

    if outpath and debug:
        print '\n' + 'saving final debug data'

        # timings
        pth = os.path.join(outfold, outfil+'_debug_timings.json')
        with open(pth, 'w') as writer:
            json.dump(info['timings'], writer)



    ##############
    # Return results metadata

    return info




    
