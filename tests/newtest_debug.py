
import automap as mapfit

# image
render = mapfit.debug.render_text_recognition('testmaps/burkina.jpg',
                                              'testmaps/burkina_georeferenced.tif')
render.save('testdebugimage.png')

# georef
render = mapfit.debug.render_georeferencing('testmaps/burkina_georeferenced.tif')
render.save('testdebuggeoref.png')
