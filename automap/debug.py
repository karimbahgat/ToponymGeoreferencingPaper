
import pythongis as pg
from PIL import Image
import numpy as np
import json
import os

import automap as mapfit

# Debugging basic processing

def render_text_recognition(imagepath, georefpath):
    # load image
    print 'loading'
    im = Image.open(imagepath)
    imagedata = pg.RasterData(image=im)
    
    # init renderer
    render = pg.renderer.Map(width=imagedata.width, height=imagedata.height, background='white')
    render._create_drawer()
    render.drawer.pixel_space()

    # determine paths (all debug files are relative to this)
    georef_root,ext = os.path.splitext(georefpath)
    segmentpath = georef_root + '_debug_segmentation.geojson'
    textpath = georef_root + '_debug_text.geojson'
    toponympath = georef_root + '_debug_text_toponyms.geojson'
    gcppath = georef_root + '_controlpoints.geojson'

    # add image
    render.add_layer(imagedata, transparency=0.5)

    # add image regions
    try:
        segmentdata = pg.VectorData(segmentpath)
        render.add_layer(segmentdata, fillcolor=None, outlinecolor='red', outlinewidth=0.3)
    except:
        # segment file is empty feature collection
        pass

    # add text
    textdata = pg.VectorData(textpath)
    def gettopleft(f):
        xmin,ymin,xmax,ymax = f.bbox
        return xmin,ymin
    def getbbox(f):
        xmin,ymin,xmax,ymax = f.bbox
        w = xmax-xmin
        h = ymax-ymin
        return xmin,ymin-h*2,xmax+xmax*2,ymax-h
##    render.add_layer(textdata, text=lambda f: f['text'],
##                     textoptions={'textcolor':'green', 'xy':gettopleft, 'anchor':'sw', 'textsize':6},
##                     fillcolor=None, outlinecolor='green')
    for col,coltextdata in textdata.manage.split('color'):
        render.add_layer(coltextdata, text=lambda f: f['text'],
                         textoptions={'textcolor':col, 'xy':gettopleft, 'anchor':'sw', 'textsize':6},
                         fillcolor=None, outlinecolor=col, outlinewidth='2px')

    # add toponym anchors
    toponymdata = pg.VectorData(toponympath)
    render.add_layer(toponymdata, fillsize=3, fillcolor=None, outlinecolor='blue', outlinewidth=0.3)

    # add final tiepoints
    gcpdata = pg.VectorData(gcppath)
    for f in gcpdata:
        col,row = f['origx'],f['origy']
        geoj = {'type':'Point', 'coordinates':(col,row)}
        f.geometry = geoj
    if len(gcpdata):
        render.add_layer(gcpdata, fillsize=3, fillcolor=(255,0,0,200), outlinecolor=None)

    # view
    render.zoom_auto()
    return render

def render_georeferencing(georefpath):
    # init renderer
    render = pg.renderer.Map(3000,3000, background='white')

    # add country background
    render.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp",
                         fillcolor=(222,222,222))

    # load georeferenced image
    print 'loading'
    georefdata = pg.RasterData(georefpath)
    georefdata.mask = georefdata.bands[-1].compute('255-val').img # use alpha band as mask
    render.add_layer(georefdata, transparency=0.3)

    # determine paths (all debug files are relative to this)
    georef_root,ext = os.path.splitext(georefpath)
    candidatepath = georef_root + '_debug_gcps_matched.geojson'
    gcppath = georef_root + '_controlpoints.geojson'
    transpath = georef_root + '_transform.json'

    # add gcps as marked in image pixels, ie calculate using the transform...
    with open(transpath) as fobj:
        transinfo = json.load(fobj)
    forward = mapfit.transforms.from_json(transinfo['forward']['model'])
    pixdata = pg.VectorData(gcppath)
    for f in pixdata:
        px,py = f['origx'],f['origy']
        x,y = forward.predict([px],[py])
        f.geometry = {'type':'Point', 'coordinates':(x,y)}
    render.add_layer(pixdata, fillsize=1, fillcolor='red',
                     text=lambda f: f['origname'],
                     textoptions={'textcolor':'darkred', 'anchor':'w', 'textsize':8})

    # add arrow from pixel to gcp
    linedata = pg.VectorData()
    for f in pixdata:
        x1,y1 = f.geometry['coordinates']
        x2,y2 = f['matchx'],f['matchy']
        geoj = {'type':'LineString', 'coordinates':[(x1,y1),(x2,y2)]}
        linedata.add_feature([], geoj)
    render.add_layer(linedata, fillsize=0.2, fillcolor='black')

    # add final controlpoints used to estimate transform
    gcpdata = pg.VectorData(gcppath)
    render.add_layer(gcpdata, fillsize=1, fillcolor='green',
                     text=lambda f: f['matchname'].split('|')[0],
                     textoptions={'textcolor':'darkgreen', 'anchor':'w', 'textsize':8})

    # view
    render.zoom_bbox(*georefdata.bbox)
    render.zoom_out(1.2)
    return render

# Error calculation and surfaces if true coords are known

def sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type='geographic'):
    # two different coordsys in and out,
    # ...comparing three different transforms between in and out (two rasters and an arbitrary transform)
    # in = pixels
    # out = coords
    # TT = truth image transform (affine)
    # ET = estimated image transform (affine)
    # ET2 = alternative estimated transform (arbitrary form)

    # ST_out sample coords are given
    # the *_in pixel coordsys cannot be assumed to be equivalent
    
    # TT_in -> TT -> TT_out      (truth_forward and truth_backward)
    # ET_in -> ET -> ET_out      (not used I guess...)
    # ET2_in -> ET2 -> ET2_out   (sampling_trans = ET2-1)

    # guiding rule is that input coordsys are different, so always use output coordsys as starting point
    # ie strategy is to send sample point through arbitrary transform ET2 and then back through TT to get the error of ET2

    # args:
    # - ST_out sample coords
    # - truth rast providing TT and TT-1
    # - trans providing ET2-1

    if error_type == 'geographic':

        # geographic error
        # (goal: diff bw ET2 and TT)
        # 1) ST_out -> ET2-1 -> ET2_in   (pixel where coord was sampled)
        # 2) ET2_in -> TT -> TT_out      (coord that truly belongs to pixel)
        # 3) voila, calc distances() bw ST_out and TT_out

        sample_pxs,sample_pys = sampling_trans.predict(sample_xs, sample_ys)
        truth_xs,truth_ys = truth_forward.predict(sample_pxs, sample_pys)
        resids = mapfit.accuracy.distances(truth_xs, truth_ys, sample_xs, sample_ys, 'geodesic')
        return sample_xs, sample_ys, truth_xs, truth_ys, resids

    elif error_type == 'pixel':

        # pixel error
        # (goal: diff bw ET2 and TT)
        # 1) ST_out -> ET2-1 -> ET2_in   (pixel where coord was sampled)
        # 2) ST_out -> TT-1 -> TT_in     (pixel that truly belongs to coord)
        # 4) voila, calc distances() bw ET2_in and TT_in

        sample_pxs,sample_pys = sampling_trans.predict(sample_xs, sample_ys)
        truth_pxs,truth_pys = truth_backward.predict(sample_xs, sample_ys)
        resids = mapfit.accuracy.distances(truth_pxs, truth_pys, sample_pxs, sample_pys, 'eucledian')
        return sample_pxs, sample_pys, truth_pxs, truth_pys, resids

    else:
        raise ValueError(str(error_type))

def error_surface_georef(sampling_trans, georef, truth, error_type='geographic'):
    print('Calculating {} error surface for georef'.format(error_type))
    ### georef grid
    # ST_out sampled from georef grid
    out = georef.copy(shallow=True)
    out.mode = 'float32'

    # prep args
    #samples = [out.cell_to_geo(col,row) for row in range(out.height) for col in range(out.width)]
    #sample_xs,sample_ys = zip(*samples)
    A = np.eye(3).flatten()
    A[:6] = list(out.affine)
    out_forward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))
    cols,rows = zip(*[(col,row) for row in range(out.height) for col in range(out.width)])
    sample_xs,sample_ys = out_forward.predict(cols, rows)

    # convert sample coords to wgs84 if necessary
    if 'longlat' not in out.crs.to_proj4():
        fromcrs = out.crs.to_proj4()
        tocrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        trans = mapfit.transforms.Projection(fromcrs, tocrs) # sample coord to wgs84
        sample_xs,sample_ys = trans.predict(sample_xs, sample_ys)

    # also chain sampling_trans if georef is not wgs84
    # for now ok, georef is always wgs84
    # ...

    # truth pixel to truth coord
    A = np.eye(3).flatten()
    A[:6] = list(truth.affine)
    truth_forward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))

    # truth coord to truth pixel
    A = np.eye(3).flatten()
    A[:6] = list(truth.inv_affine)
    truth_backward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))

    # add conversion between truth crs and wgs84 if necessary
    if 'longlat' not in truth.crs.to_proj4():
        # truth coord to wgs84
        fromcrs = truth.crs.to_proj4()
        tocrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        trans1 = truth_forward # pixel to truth coord
        trans2 = mapfit.transforms.Projection(fromcrs, tocrs) # truth coord to wgs84
        truth_forward = mapfit.transforms.Chain([trans1, trans2])

        # wgs84 to truth coord
        fromcrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        tocrs = truth.crs.to_proj4()
        trans1 = mapfit.transforms.Projection(fromcrs, tocrs) # wgs84 to truth coord
        trans2 = truth_backward # truth coord to pixel
        truth_backward = mapfit.transforms.Chain([trans1, trans2])

    # calc errors
    sample_xs,sample_ys,truth_xs,truth_ys,errors = sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type=error_type)

    # return as raster
    errors = errors.reshape((out.height,out.width))
    out.add_band(img=Image.fromarray(errors))

    # maybe blank out nodata values from the error surface?
    #out.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    
    return out

def error_surface_image(sampling_trans, georef, truth, error_type='pixel'):
    print('Calculating {} error surface for image'.format(error_type))
    ### image grid
    # ST_out sampled from image grid
    # otherwise, same exact steps
    out = truth.copy(shallow=True)
    out.mode = 'float32'

    # prep args
    #samples = [out.cell_to_geo(col,row) for row in range(out.height) for col in range(out.width)]
    #sample_xs,sample_ys = zip(*samples)
    A = np.eye(3).flatten()
    A[:6] = list(out.affine)
    out_forward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))
    cols,rows = zip(*[(col,row) for row in range(out.height) for col in range(out.width)])
    sample_xs,sample_ys = out_forward.predict(cols, rows)

    # convert sample coords to wgs84 if necessary
    if 'longlat' not in out.crs.to_proj4():
        fromcrs = out.crs.to_proj4()
        tocrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        trans = mapfit.transforms.Projection(fromcrs, tocrs) # sample coord to wgs84
        sample_xs,sample_ys = trans.predict(sample_xs, sample_ys)

    # also chain sampling_trans if georef is not wgs84
    # for now ok, georef is always wgs84
    # ...
    
    # truth pixel to truth coord
    A = np.eye(3).flatten()
    A[:6] = list(truth.affine)
    truth_forward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))

    # truth coord to truth pixel
    A = np.eye(3).flatten()
    A[:6] = list(truth.inv_affine)
    truth_backward = mapfit.transforms.Polynomial(A=A.reshape((3,3)))

    # add conversion between truth crs and wgs84 if necessary
    if 'longlat' not in truth.crs.to_proj4():
        # truth coord to wgs84
        fromcrs = truth.crs.to_proj4()
        tocrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        trans1 = truth_forward # pixel to truth coord
        trans2 = mapfit.transforms.Projection(fromcrs, tocrs) # truth coord to wgs84
        truth_forward = mapfit.transforms.Chain([trans1, trans2])

        # wgs84 to truth coord
        fromcrs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        tocrs = truth.crs.to_proj4()
        trans1 = mapfit.transforms.Projection(fromcrs, tocrs) # wgs84 to truth coord
        trans2 = truth_backward # truth coord to pixel
        truth_backward = mapfit.transforms.Chain([trans1, trans2])

    # calc errors
    sample_xs,sample_ys,truth_xs,truth_ys,errors = sampling_errors(sample_xs, sample_ys, sampling_trans, truth_forward, truth_backward, error_type=error_type)

    # return as raster
    errors = errors.reshape((out.height,out.width))
    out.add_band(img=Image.fromarray(errors))
    return out

def error_surface_vis(rast, error_surface):
    print('Visualizing rast overlaid with errors')
    m = pg.renderer.Map(background='white',
                        #textoptions={'font':'freesans'},
                        crs=rast.crs,
                        )
    m.add_layer(rast)
    m.add_layer(error_surface, transparency=0.2, legendoptions={'title':'Error', 'valueformat':'.1f'})
    m.zoom_bbox(*rast.bbox)
    m.add_legend({'padding':0})
    m.render_all()
    return m







        
