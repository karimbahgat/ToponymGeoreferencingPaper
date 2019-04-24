
import pythongis as pg
import os

m = pg.renderer.Map(3000, 1500)
m.add_layer(r"C:\Users\kimok\Downloads\cshapes\cshapes.shp")

for fil in os.listdir('testmaps'):
    if 'ussia' in fil: continue
    if fil.endswith('_georeferenced.tif'):
        print fil
        d = pg.RasterData('testmaps/%s' % fil)
        for b in d.bands:
            b.nodataval = 0
        m.add_layer(d)

#m.view()
m.save('testmaps/_all_georefs.png')
