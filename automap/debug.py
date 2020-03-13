
import pythongis as pg
import json
import os

import automap as mapfit


def render_text_recognition(imagepath, georefpath):
    # load image
    print 'loading'
    imagedata = pg.RasterData(imagepath)
    
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

    






        
