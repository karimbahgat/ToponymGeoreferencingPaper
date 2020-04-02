
import os
import json
import automap as mapfit
import pythongis as pg
import pycrs

def inspect_image(fil, outfil):
    # image
    render = mapfit.debug.render_image_output(fil,
                                              outfil)
    render.render_all()
    return render.img

def inspect_georef(outfil):
    # georef
    render = mapfit.debug.render_georeferencing_output(outfil)
    render.render_all()
    return render.img

def inspect_image_errors(georef_fil, truth_fil, error_type):
    # image errors
    mapp = mapfit.debug.render_image_errors(georef_fil, truth_fil, error_type)
    mapp.render_all()
    return mapp.img

def inspect_georef_errors(georef_fil, truth_fil, error_type):
    # georef errors
    mapp = mapfit.debug.render_georeferencing_errors(georef_fil, truth_fil, error_type)
    mapp.render_all()
    return mapp.img

if __name__ == '__main__':
    #root = 'sim_1_1_1_image.png'
    root = 'sim_20_1_1_image.png'
    #root = 'sim_1_11_2_image.jpg'
    #root = 'sim_1_10_1_image.png'
    #root = 'sim_1_10_2_image.jpg'
    fil = 'maps/{}'.format(root)
    outfil_root = os.path.splitext(root)[0].replace('_image','')
    outfil = 'output/{}_georeferenced_auto.tif'.format(outfil_root)
    
##    inspect_image(fil, outfil).show()
##    
##    inspect_georef(outfil).show()
    
    ### 
    
    inspect_image_errors(outfil, fil, 'geographic').show()
    
    inspect_image_errors(outfil, fil, 'pixel').show()
    
    inspect_georef_errors(outfil, fil, 'geographic').show()
    
    inspect_georef_errors(outfil, fil, 'pixel').show()
    
    # and then for exact...
    #outfil = 'output/{}_georeferenced_exact.tif'.format(outfil_root)
    #inspect_georef(outfil).show()


    
