
import automap as mapfit

# first produce
db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"
info = mapfit.automap('testmaps/burkina.jpg', textcolor=None, warp_order=None, db=db, debug=True)

# image
render = mapfit.debug.render_text_recognition('testmaps/burkina.jpg',
                                              'testmaps/burkina_georeferenced.tif')
render.save('testdebugimage.png')

# georef
render = mapfit.debug.render_georeferencing('testmaps/burkina_georeferenced.tif')
render.save('testdebuggeoref.png')






