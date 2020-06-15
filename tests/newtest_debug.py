
import automap as mapfit
import os

#testim = 'burkina.jpg'
#testim = 'china_pol96.jpg'
testim = 'cameroon_pol98.jpg'
#testim = 'brazil_land_1977.jpg'
#testim = 'france_admin91.jpg'
#testim = 'russia_autonomous92.jpg'
#testim = 'israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
#testim = 'korean_war_april_1951.jpg'
testim_root,ext = os.path.splitext(testim)

# test ocr refine
##from PIL import Image
##import numpy as np
##textline = {u'word_num': u'2', 'fontheight': 33, u'line_num': u'1', u'text': u'Boba', u'top': 1288, u'level': u'5', u'page_num': u'1', u'block_num': u'130', u'height': 33, u'width': 58, u'conf': 77.0, u'par_num': u'1', u'left': 264}
##im = Image.open('testmaps/'+testim)
##im = im.resize((im.size[0]*2,im.size[1]*2), Image.LANCZOS)
####for k in 'left top width height'.split():
####    textline[k] /= 2 # downscale
##col = (0,0,0) 
##thresh = 25 
##diff = mapfit.segmentation.color_difference(mapfit.segmentation.quantize(im), col)
##diff[diff>thresh] = 255
##refined = mapfit.textdetect.refine_textbox(diff, textline)
##print refined
##fsfsd

# test thresh
##from PIL import Image
##import numpy as np
##im = Image.open('testmaps/'+testim)
##col = (188,0,0) #(33.42117154811716, 42.028535564853556, 18.39589958158996) #(26.159759905047157, 44.41302692450912, 26.73344983188299)
##thresh = 25 #18.8*1.5 #9.188087506560446+5.054860732095406 #19.892093984040308+13.6/5.0
##diff = mapfit.segmentation.color_difference(mapfit.segmentation.quantize(im),
##                                            col)
##diff[diff>thresh] = 255
##Image.fromarray(diff).show()
##fdsfds

# test closest thresh
##from PIL import Image
##import numpy as np
##im = Image.open('testmaps/'+testim)
###im = im.resize((im.size[0]*2,im.size[1]*2), Image.LANCZOS)
##im = mapfit.segmentation.quantize(im)
##colors = [(38.876272094268884, 29.178200321371186, 13.807873594001071),
##          (103.13357478782636, 72.47267419860533, 44.70322720147035)]
##colors = mapfit.textdetect.sniff_text_colors(im).keys()
##mapfit.segmentation.view_colors(colors)
##thresh = 25 
##out = np.zeros((im.size[1],im.size[0],3,), dtype=np.uint8)
##out[:,:,None] = (255,255,255)
##cumdiff = np.ones((im.size[1]*im.size[0],)) * 255
##for col in colors:
##    print col
##    diff = mapfit.segmentation.color_difference(im, col)
##    diff[diff>thresh] = 255
##    smaller = diff.flatten()<cumdiff
##    cumdiff[smaller] = diff.flatten()[smaller]
##    out[smaller.reshape((im.size[1],im.size[0]))] = map(int, col)
##Image.fromarray(out).show()
##for col in colors:
##    col = map(int, col)
##    wherecol = (out == col).all(axis=2).reshape(out.shape[:2])
##    colarr = np.ones(out.shape, dtype=np.uint8) * 255
##    colarr[wherecol] = col
##    Image.fromarray(colarr).show()
##fdsfds

# first produce
db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"
info = mapfit.automap('testmaps/{}'.format(testim), textcolor=None, warp_order=None, db=db, debug=True)

# image
render = mapfit.debug.render_image_output('testmaps/{}'.format(testim),
                                              'testmaps/{}_georeferenced.tif'.format(testim_root))
render.save('testdebugimage.png')

# georef
render = mapfit.debug.render_georeferencing_output('testmaps/{}_georeferenced.tif'.format(testim_root))
render.save('testdebuggeoref.png')






