
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

import time
import os
import json



def automap(inpath, outpath=True, matchthresh=0.1, textcolor=None, colorthresh=25, textconf=60, sample=False, db=None, source='gns', warp_order=None, residual_type='pixels', max_residual=None, debug=False, **kwargs):
    start = time.time()
    info = dict()
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
            outfil = infil
        else:
            # relative to manual outpath
            outfold,outfil = os.path.split(outpath)
            outfil,ext = os.path.splitext(outfil)
    else:
        # dont output, but still need for debug
        # relative to inpath
        outfold = infold
        outfil = infil
    




    ################
    # Image partitioning
    
    # partition image
    print 'image segmentation'
    mapp_poly,box_polys = segmentation.image_segments(im)

    # create as feature collection (move to image_segments()?)
    seginfo = {'type': 'FeatureCollection',
               'features': []}
    
    # (map)
    mapp_geoj = {'type': 'Polygon',
                 'coordinates': [ [tuple(p[0].tolist()) for p in mapp_poly] ]}
    props = {'type':'Map'}
    feat = {'type': 'Feature', 'properties': props, 'geometry': mapp_geoj}
    seginfo['features'].append(feat)
    
    # (boxes)
    boxes_geoj = [{'type': 'Polygon',
                 'coordinates': [ [tuple(p[0].tolist()) for p in box] ]}
                  for box in box_polys]
    for box_geoj in boxes_geoj:
        props = {'type':'Box'}
        feat = {'type': 'Feature', 'properties': props, 'geometry': box_geoj}
        seginfo['features'].append(feat)

    # store metadata
    info['segmentation'] = seginfo
    



    ###############
    # Text detection

    # remove unwanted parts of image
    text_im = im
    if mapp_poly is not None:
        text_im = segmentation.mask_image(text_im, mapp_poly)
    for box in box_polys:
        text_im = segmentation.mask_image(text_im, box, invert=True)
    #text_im.show()
    
    # detect text
    print 'detecting text'
    if textcolor and not isinstance(textcolor, list):
        textcolor = [textcolor]
    texts = textdetect.auto_detect_text(text_im, textcolors=textcolor, colorthresh=colorthresh, textconf=textconf, sample=sample)
    toponym_colors = set((r['color'] for r in texts))

    # deduplicate overlapping texts from different colors
    # ...

    # connect text
    print '(connecting texts)'
    grouped = []
    for col in toponym_colors:
        coltexts = [r for r in texts if r['color'] == col]
        grouped.extend( textgroup.connect_text(coltexts) )
    texts = grouped

    # ignore small texts?
    texts = [text for text in texts if len(text['text_clean']) >= 3]

    # store metadata
    textinfo = {'type': 'FeatureCollection', 'features': []}
    for r in texts:
        x1,y1,x2,y2 = r['left'], r['top'], r['left']+r['width'], r['top']+r['height']
        box = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
        geoj = {'type':'Polygon', 'coordinates':[box]}
        props = dict(r)
        feat = {'geometry':geoj, 'properties':props}
        textinfo['features'].append(feat)
    info['text_recognition'] = textinfo




    ################
    # Toponym selection

    # text anchor points
    print 'determening text anchors'
    anchored = []
    for col in toponym_colors:
        coltexts = [r for r in texts if r['color'] == col]
        diff = segmentation.color_difference(segmentation.quantize(im), col)
        diff[diff > colorthresh] = 255
        anchor_im = PIL.Image.fromarray(diff)
        anchored.extend( toponyms.detect_toponym_anchors(anchor_im, coltexts) )
    texts = anchored

    # create control points from toponyms
    points = [(r['text_clean'], r['anchor']) for r in texts if 'anchor' in r] # if r['function']=='placename']

    # store metadata
    toponyminfo = {'type': 'FeatureCollection', 'features': []}
    for name,p in points:
        geoj = {'type':'Point', 'coordinates':p}
        props = {'name':name}
        feat = {'geometry':geoj, 'properties':props}
        toponyminfo['features'].append(feat)
    info['toponym_candidates'] = toponyminfo

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)

    


    ###############
    # Control point matching

    # find matches
    print 'finding matches'
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
    info['gcps_matched'] = gcps_matched_info

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)




    #################
    # Transformation

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
    trans, pixels, coords, err, resids = accuracy.auto_drop_models(trans, pixels, coords,
                                                                 improvement_ratio=0.10,
                                                                 minpoints=None,
                                                                 leave_one_out=True,
                                                                 invert=invert, distance=distance,
                                                                 accuracy='rmse')
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
    info['gcps_final'] = gcps_final_info

    # then transforms
    # NOTE: metadata reports only RMSE error type with leave_one_out=True, maybe allow user customizing this? 
    if invert:
        ferr,fresids = err,resids # already calculated
        berr,bresids = accuracy.model_accuracy(trans, pixels, coords,
                                             leave_one_out=True,
                                             invert=True, distance='geodesic',
                                             accuracy='rmse')
    else:
        berr,bresids = err,resids # already calculated
        ferr,fresids = accuracy.model_accuracy(trans, pixels, coords,
                                             leave_one_out=True,
                                             invert=False, distance='eucledian',
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
    info['transformation'] = {'forward': forward_info,
                              'backward': backward_info}



    #################
    # Warping

    # warp the image
    print '\n'+'warping'
    print '{} points, warp_method={}'.format(len(tiepoints), forward)
    if mapp_poly is not None:
        mapp_im = segmentation.mask_image(im.convert('RGBA'), mapp_poly) # map region
    else:
        mapp_im = im
    wim,aff = imwarp.warp(mapp_im, forward, backward) # warp

    # store metadata
    warp_info = {'image':wim,
                 'affine':aff}
    info['warped'] = warp_info




    ##############
    # Save output? 

    if outpath:
        
        # warped image
        pth = os.path.join(outfold, outfil + '_georeferenced.tif')
        rast = pg.RasterData(image=wim, affine=aff) # to geodata
        rast.save(pth)

        # final control points
        pth = os.path.join(outfold, outfil+'_controlpoints.geojson')
        with open(pth, 'w') as writer:
            json.dump(info['gcps_final'], writer)

        # transformation
        pth = os.path.join(outfold, outfil+'_transform.json')
        with open(pth, 'w') as writer:
            json.dump(info['transformation'], writer)

    if debug:

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



    #############
    # Finished

    print '\n'+'finished!'
    print 'total runtime: {:.1f} seconds \n'.format(time.time() - start)

    return info



    #################

##    def debug_ocr(im, outpath, data, controlpoints, origs):
##        import pyagg
##        c = pyagg.canvas.from_image(im)
##        #print c.width,c.height,im.size
##        for r in data:
##            top,left,w,h = [r[k] for k in 'top left width height'.split()]
##            box = [left, top, left+w, top+h]
##            text = r.get('text_clean','[?]')
##            #print box,text
##            c.draw_box(bbox=box, fillcolor=None, outlinecolor=(0,255,0))
##            c.draw_text(text, xy=(left,top), anchor='sw', textsize=6, textcolor=(0,255,0)) #bbox=box)
##        for oname,ocoord in origs:
##            c.draw_circle(xy=ocoord, fillsize=1, fillcolor=None, outlinecolor=(0,0,255))
##        for on,oc,mn,mc,res in controlpoints:
##            c.draw_circle(xy=oc, fillsize=1, fillcolor=(255,0,0,155), outlinecolor=None)
##        c.save(outpath)
##
##    def debug_warped(pth, outpath, controlpoints):
##        import pythongis as pg
##        m = pg.renderer.Map(width=2000, height=2000, background='white')
##
##        m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_admin_0_countries\ne_10m_admin_0_countries.shp",
##                    fillcolor=(217,156,38))
##
##        warped = pg.RasterData(pth)
##        for b in warped.bands:
##            b.nodataval = 0 # need better approach, use 4th band as mask
##        rlyr = m.add_layer(warped, transparency=0.3)
##
##        m.add_layer(r"C:\Users\kimok\Downloads\ne_10m_populated_places_simple\ne_10m_populated_places_simple.shp",
##                    fillcolor='red', fillsize=0.1) #outlinewidth=0.1)
##
##        anchors = pg.VectorData(fields=['origname', 'matchname', 'residual'])
##        for on,oc,mn,mc,res in controlpoints:
##            anchors.add_feature([on,mn,res], dict(type='Point', coordinates=mc))
##        m.add_layer(anchors, fillcolor=(0,255,0), fillsize=0.3)
##
##        m.zoom_bbox(*rlyr.bbox)
##        m.zoom_out(1.5)
##        #m.view()
##        m.save(outpath)

    ##    # draw data onto image
##    if debug is True or debug == 'ocr':
##        debugpath = os.path.join(outfold, outfil+'_debug_ocr.png')
##        debug_ocr(im, debugpath, data, controlpoints, origs)
##
##    # view warped
##    if debug is True:
##        debugpath = os.path.join(outfold, outfil+'_debug_warp.png')
##        debug_warped(outpath, debugpath, controlpoints)

    
