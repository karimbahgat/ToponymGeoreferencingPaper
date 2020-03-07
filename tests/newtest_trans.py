
import automap as mapfit
import json

with open('testmaps/china_pol96_georeferenced_transform.json') as fobj:
    transinfo = json.load(fobj)

forw = mapfit.transforms.from_json(transinfo['forward']['model'])
print forw
backw = mapfit.transforms.from_json(transinfo['backward']['model'])
print backw

px,py = 0,0 #958,429 #1000,1000
print 'from',px,py
x,y = forw.predict([px], [py])
print 'to',x,y
px2,py2 = backw.predict(x, y)
print 'and back',px2,py2

import numpy as np
newback = mapfit.transforms.Polynomial(A=forw.A)
newback.A = np.linalg.inv(newback.A)
px2,py2 = newback.predict(x, y)
print 'inverted back',px2,py2

### 

with open('testmaps/china_pol96_georeferenced_controlpoints.geojson') as fobj:
    gcps = json.load(fobj)
inx = [f['properties']['origx'] for f in gcps['features']]
iny = [f['properties']['origy'] for f in gcps['features']]
outx = [f['properties']['matchx'] for f in gcps['features']]
outy = [f['properties']['matchy'] for f in gcps['features']]
forw = mapfit.transforms.Polynomial(order=2)
forw.fit(inx,iny,outx,outy)
backw = mapfit.transforms.Polynomial(order=2)
backw.fit(inx,iny,outx,outy, invert=True)

px,py = 0,0 #958,429 #666,409 #1000,1000
print 'from',px,py
x,y = forw.predict([px], [py])
print 'to',x,y
px2,py2 = backw.predict(x, y)
print 'and back',px2,py2
##predx,predy = forw.predict(inx, iny)
##backw = mapfit.transforms.Polynomial(order=2)
##backw.fit(predx,predy,inx,iny)
##px2,py2 = backw.predict(x, y)
##print 'and back again',px2,py2



