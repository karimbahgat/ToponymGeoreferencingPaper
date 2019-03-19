
import pythongis as pg
import os

m = pg.renderer.Map(1000, 500)
m.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp")

for fil in os.listdir('testmaps'):
    if fil.endswith('_georeferenced.tif'):
        print fil
        d = pg.RasterData('testmaps/%s' % fil)
        m.add_layer(d)

m.view()
