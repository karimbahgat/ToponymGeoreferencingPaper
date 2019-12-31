
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



def automap(inpath, outpath=None, matchthresh=0.1, textcolor=None, colorthresh=25, textconf=60, sample=False, db=None, source='gns', warp_order=None, max_residual=None, debug=False, **kwargs):
    start = time.time()
    
    print 'loading image', inpath
    im = PIL.Image.open(inpath).convert('RGB')

    # determine various paths
    infold,infil = os.path.split(inpath)
    infil,ext = os.path.splitext(infil)
    if not outpath:
        outpath = os.path.join(infold, infil + '_georeferenced.tif')
    outfold,outfil = os.path.split(outpath)
    outfil,ext = os.path.splitext(outfil)



    ################
    
    # partition image
    print 'image segmentation'
    mapp_poly,box_polys = segmentation.image_segments(im)



    ###############

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

    # connect text
    print '(connecting texts)'
    grouped = []
    for col in toponym_colors:
        coltexts = [r for r in texts if r['color'] == col]
        grouped.extend( textgroup.connect_text(coltexts) )
    texts = grouped

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

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)

    

    ###############

    # find matches
    print 'finding matches'
    origs,matches = triangulate.find_matches(points, matchthresh, db=db, source=source, **kwargs)
    orignames,origcoords = zip(*origs)
    matchnames,matchcoords = zip(*matches)
    tiepoints = zip(origcoords, matchcoords)

    print '\n'+'time so far: {:.1f} seconds \n'.format(time.time() - start)



    #################

    def estimate_polynomial(tiepoints, order):
        pixels,coords = zip(*tiepoints)
        (cols,rows),(xs,ys) = zip(*pixels),zip(*coords)
        forward = transforms.Polynomial(order=order)
        forward.fit(cols,rows,xs,ys)
        backward = transforms.Polynomial(order=order)
        backward.fit(cols,rows,xs,ys, invert=True)
        return forward,backward

    def get_rmse(transform, inpoints, outpoints):
        inx,iny = zip(*inpoints)
        outx,outy = zip(*outpoints)
        predx,predy = backward.predict(inx, iny)
        resids = accuracy.residuals(outx,outy,predx,predy)
        rmse = accuracy.RMSE(resids)
        return rmse, resids

    # estimate transforms
    forward,backward = estimate_polynomial(tiepoints, warp_order)

    # initial rmse
    frompoints,topoints = zip(*tiepoints)
    rmse,resids = get_rmse(backward, topoints, frompoints)
    print '{} points, RMSE: {}'.format(len(frompoints), rmse)

    # drop outliers (in terms of pixels, flipping from/topoints)
    print '\n'+'excluding outliers'
    if max_residual == 'auto':
        maxres = max(im.size[0], im.size[1]) / 100.0 * 10 # not more than 10 percent of image
    else:
        maxres = None
    topoints,frompoints = accuracy.drop_outliers(backward, topoints, frompoints, max_residual=maxres)
    tiepoints = zip(frompoints, topoints)

    # reestimate transforms
    forward,backward = estimate_polynomial(tiepoints, warp_order)

    # calculate new rmse
    rmse,resids = get_rmse(backward, topoints, frompoints)
    print '{} points, RMSE: {}'.format(len(frompoints), rmse)



    #################

    def final_controlpoints(tiepoints, residuals, origs, matches, outpath=False):
        controlpoints = []
        orignames,origcoords = zip(*origs)
        matchnames,matchcoords = zip(*matches)
        for (oc,mc),res in zip(tiepoints, residuals):
            i = origcoords.index(oc)
            oc,on,mc,mn = [lst[i] for lst in (origcoords,orignames,matchcoords,matchnames)]
            controlpoints.append([on,oc,mn,mc,res])

        if outpath:
            vec = pg.VectorData(fields='origname origx origy matchname matchx matchy residual'.split())
            for on,(ox,oy),mn,(mx,my),res in controlpoints:
                vec.add_feature([on,ox,oy,mn,mx,my,res], geometry={'type': 'Point', 'coordinates': (mx,my)})
            vec.save(outpath)

        return controlpoints

    # warp
    print '\n'+'warping'
    print '{} points, warp_method={}'.format(len(tiepoints), forward.order)
    if mapp_poly is not None:
        mapp_im = segmentation.mask_image(im.convert('RGBA'), mapp_poly) # map region
    else:
        mapp_im = im
    wim,aff = imwarp.warp(mapp_im, forward, backward) # warp
    rast = pg.RasterData(image=wim, affine=aff) # to geodata
    if outpath:
        rast.save(outpath)

    # final control points
    cppath = os.path.join(outfold, outfil+'_controlpoints.geojson')
    controlpoints = final_controlpoints(tiepoints, resids, origs, matches, outpath=cppath)

    print '\n'+'finished!'
    print 'total runtime: {:.1f} seconds \n'.format(time.time() - start)



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

    
