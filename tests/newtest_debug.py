
import automap as mapfit
import os

#testim = 'brazil_land_1977.jpg'
testim = 'china_pol96.jpg'
testim_root,ext = os.path.splitext(testim)

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






