
import pythongis as pg
import os


def render_text_recognition(imagepath, georefpath):
    # load image
    print 'loading'
    imagedata = pg.RasterData(imagepath)
    
    # init renderer
    render = pg.renderer.Map(width=imagedata.width, height=imagedata.height)
    render._create_drawer()
    render.drawer.pixel_space()

    # determine paths (all debug files are relative to this)
    georef_root,ext = os.path.splitext(georefpath)
    segmentpath = georef_root + '_debug_segmentation.geojson'
    textpath = georef_root + '_debug_text.geojson'
    toponympath = georef_root + '_debug_text_toponyms.geojson'
    gcppath = georef_root + '_controlpoints.geojson'

    # add image
    render.add_layer(imagedata)

    # add image regions
    segmentdata = pg.VectorData(segmentpath)
    render.add_layer(segmentdata, fillcolor=None, outlinecolor='red', outlinewidth=0.3)

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
    render.add_layer(textdata, text=lambda f: f['text'],
                     textoptions={'textcolor':'green', 'xy':gettopleft, 'anchor':'sw', 'textsize':6},
                     fillcolor=None, outlinecolor='green')

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
    render = pg.renderer.Map(3000,3000)

    # add country background
    render.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp",
                         fillcolor=(222,222,222))

    # load georeferenced image
    print 'loading'
    georefdata = pg.RasterData(georefpath) 
    render.add_layer(georefdata, transparency=0.3)

    # determine paths (all debug files are relative to this)
    georef_root,ext = os.path.splitext(georefpath)
    candidatepath = georef_root + '_debug_gcps_matched.geojson'
    gcppath = georef_root + '_controlpoints.geojson'

    # add gcp candidates
    candidatedata = pg.VectorData(candidatepath)
    render.add_layer(candidatedata, fillsize=1, fillcolor=None, outlinecolor='blue')

    # add final controlpoints
    gcpdata = pg.VectorData(gcppath)
    render.add_layer(gcpdata, fillsize=1, fillcolor='green')

    # TODO: maybe add where gcps actually end up, eg calculate using the transform...
    # ... 

    # view
    render.zoom_bbox(*georefdata.bbox)
    render.zoom_out(1.2)
    return render

    






        
