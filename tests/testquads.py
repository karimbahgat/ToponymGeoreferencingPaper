
import automap as mapfit
import pyagg

# test raw quads
##c = pyagg.Canvas(800, 400, background='yellow')
##c.geographic_space()
##
##q = mapfit.segmentation.Quad(0, 0, 360, 180)
##q.split()
##
##for i in range(4 + 4**2):
##    samp = q.sample()
##    print samp, (samp.x,samp.y)
##    c.draw_text(str(i+1), (samp.x,samp.y))
##
##c.view()

# test bbox quad sampling
c = pyagg.Canvas(800, 400, background='yellow')
c.geographic_space()

bbox = [-180,90,180,-90]
samplesize = (180/1.5, 90/1.5)
for i,samp in enumerate(mapfit.segmentation.sample_quads(bbox, samplesize)):
    print samp, (samp.x,samp.y)
    c.draw_box(bbox=samp.bbox(), fillcolor='blue')
    c.draw_text(str(i+1), (samp.x,samp.y))

c.view()

