
import os
import json
import automap as mapfit
import pythongis as pg
import pycrs

def inspect_image(fil, outfil):
    # image
    render = mapfit.debug.render_text_recognition(fil,
                                                  outfil)
    render.render_all()
    return render.img

def inspect_georef(outfil):
    # georef
    render = mapfit.debug.render_georeferencing(outfil)
    render.render_all()
    return render.img

def true_image_errors(georef_fil, truth_fil, error_type):
    georef_root,ext = os.path.splitext(georef_fil)

    # original/simulated map
    truth = pg.RasterData(truth_fil)
##    if 'longlat' not in truth.crs.to_proj4():
##        print 'reprojecting to longlat'
##        truth = truth.manage.reproject("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    print truth.affine
    
    # georeferenced/transformed map
    georef = pg.RasterData(georef_fil)
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    print georef.affine

    # calc error surface
    with open('{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
        sampling_trans = mapfit.transforms.from_json(transdict['backward']['model'])
    image_errorsurf = mapfit.debug.error_surface_image(sampling_trans, georef, truth, error_type)

    # visualize geographic error
    mapp = mapfit.debug.error_surface_vis(truth, image_errorsurf)
    mapp.render_all()
    return mapp.img

def true_georef_errors(georef_fil, truth_fil, error_type):
    georef_root,ext = os.path.splitext(georef_fil)

    # original/simulated map
    truth = pg.RasterData(truth_fil)
##    if 'longlat' not in truth.crs.to_proj4():
##        print 'reprojecting to longlat'
##        truth = truth.manage.reproject("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    print truth.affine
    
    # georeferenced/transformed map
    georef = pg.RasterData(georef_fil)
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    print georef.affine

    # calc error surface
    with open('{}_transform.json'.format(georef_root), 'r') as fobj:
        transdict = json.load(fobj)
        sampling_trans = mapfit.transforms.from_json(transdict['backward']['model'])
    georef_errorsurf = mapfit.debug.error_surface_georef(sampling_trans, georef, truth, error_type)

    # visualize geographic error
    mapp = mapfit.debug.error_surface_vis(georef, georef_errorsurf)
    mapp.render_all()
    return mapp.img

if __name__ == '__main__':
    #root = 'sim_1_1_1_image.png'
    #root = 'sim_20_1_1_image.png'
    #root = 'sim_1_11_2_image.jpg'
    root = 'sim_1_10_1_image.png'
    #root = 'sim_1_10_2_image.jpg'
    fil = 'maps/{}'.format(root)
    outfil_root = os.path.splitext(root)[0].replace('_image','')
    outfil = 'output/{}_georeferenced_auto.tif'.format(outfil_root)
    
##    inspect_image(fil, outfil).show()
##    
##    inspect_georef(outfil).show()
    
    ### 
    
    true_image_errors(outfil, fil, 'geographic').show()
    
    true_image_errors(outfil, fil, 'pixel').show()
    
    true_georef_errors(outfil, fil, 'geographic').show()
    
    true_georef_errors(outfil, fil, 'pixel').show()
    
    # and then for exact...
    #outfil = 'output/{}_georeferenced_exact.tif'.format(outfil_root)
    #inspect_georef(outfil).show()


    
