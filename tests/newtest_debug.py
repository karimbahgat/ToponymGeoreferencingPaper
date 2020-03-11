
import automap as mapfit
import os

#testim = 'burkina.jpg'
testim = 'china_pol96.jpg'
#testim = 'brazil_land_1977.jpg'
testim_root,ext = os.path.splitext(testim)

# specific test
##from PIL import Image
##import numpy as np
##im = Image.open('testmaps/'+testim)
##col = (0,0,0) #(33.42117154811716, 42.028535564853556, 18.39589958158996) #(26.159759905047157, 44.41302692450912, 26.73344983188299)
##thresh = 15 #18.8*1.5 #9.188087506560446+5.054860732095406 #19.892093984040308+13.6/5.0
##diff = mapfit.segmentation.color_difference(mapfit.segmentation.quantize(im),
##                                            col)
##diff[diff>thresh] = 255
##Image.fromarray(diff).show()

# first produce
db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"
info = mapfit.automap('testmaps/{}'.format(testim), textcolor=None, warp_order=None, db=db, debug=True)

# image
render = mapfit.debug.render_text_recognition('testmaps/{}'.format(testim),
                                              'testmaps/{}_georeferenced.tif'.format(testim_root))
render.save('testdebugimage.png')

# georef
render = mapfit.debug.render_georeferencing('testmaps/{}_georeferenced.tif'.format(testim_root))
render.save('testdebuggeoref.png')






