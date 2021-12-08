import sys
sys.path.insert(0, '../dependencies/georeference maps')

import pythongis as pg
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math
import sqlite3

import automap as mapfit




print(os.getcwd())
try:
    os.chdir('realworld')
except:
    pass




# Loop and create table
if True:
    out = pg.VectorData()
    out.fields = ['filename', 'url', 'width', 'height', 'georef']

    for fil in os.listdir('scrape'):
        #print fil
        if fil.endswith('.json'):
            with open('scrape/'+fil) as fobj:
                dct = json.load(fobj)
                dct['filename'] = fil
                del dct['thumbnail']
        
            out.add_feature(dct, None)

    # encode georef dict to json for saving
    #print 'json dumping'
    #out.compute('georef', lambda f: json.dumps(f['georef']))

    # save
    #out.save('analyze/stats.csv')
    #db = sqlite3.connect('scrape/stats.db')
    #cur = db.cursor()
    #fielddef = ','.join(['{} text'.format(fl) for fl in out.fields])
    #cur.execute('create table data ({})'.format(fielddef))
    #for f in out:
    #  qs = ','.join('?'*len(out.fields))
    #  vals = tuple(f.row)
    #  cur.execute('insert into data values ({})'.format(qs), vals)
    #db.commit()

print(out)

# drop weird maps from the automatic scraping process 
# (pure islands, no toponyms, non-maps)

dropfiles = ['0ad520bf73b8c8906cbe31ed5ba7a7726f75473c00c337cb5e94572f.json', '15869a2cabb66f3c2d70ffd0338e83d8e7e5c4060a94f1c3a0166062.json', '1c1bf2ff40bce027f1b486a6e01b185e77a26df2a41325ebf1c0a3f1.json', '6552b0728bf127a41b89ad089783989649dfb739eadce9826ff702b3.json', '69356071f9f0d04f3c07682eaef07e55f5bc45d4325336f5b8d1fb6d.json', '7644d9dcda6d3afa8ee16d1f693a863cb0608233c97f5c302c23bd02.json', '77fee64502c3b086389b5453fa40f7abcb80a41eee689307fb43e55c.json', '782d26f76e13d29bd83ec92b811adaee926c71e2e58c2e912913d309.json', '86b774d09c6da46ae3ef9c084f23a579d2b4407ec2bcc8555afa887f.json', '8708f351b77c9762948b962db3cf2d9e15d8ee8ba7edb04063cf53b1.json', '8b0030f673beb43a852efd71c1841552e8e0bf8e6bb09803a7e63812.json', '923df4a25a50caff6d7f68559a486540c792fca3f6d545c6e1f556ad.json', 'a17f7e8c39c24e2547cc04ee628341ac16ad07c58c28b66ac79fe6db.json', 'a6ee245417147630d7fe3cb9e1ae6d088ea1dff66a7187416f19aa10.json', 'b5328a7cf7e930b241eb5b1f7a7912772a1c395bfe68f99cbc3d2f6a.json', 'bc2005bfb0188d1ce4a645540091c22f7de8979e3dac5fd8118ec75c.json', 'bf27f8edce01451c6556c175de3350884d7f0fd24884949bf40c505a.json', 'bf68323a896b424d694e4144d9857047487ab390e33986270c5f6e4b.json', 'c70730f9ac2cebb546885e2ad63a4f12bee70a53b81e42bb937393d3.json', 'cec085d33fa142d1924bd9f7541c1deaf4006a5df310d9bf630306d1.json', 'd1159a0a2bffc69dc89033bda1179a981a7d362701c31bbb6185753f.json', 'd4c4e8bdde44903724fd89eeab5d6a6471b2e21fc41e655c1f0e175d.json', 'dd8e7601355c1c9e9b0bd09690a8f609f527a1804c7e607eae47d571.json', 'ddef35f92e98f8a7a98930f474dd2299d766f2295c8a6fb34c67fbcc.json', 'df63d87fef946a3bce9324dc654f2e9fbc08024344d5780d8b045990.json', 'e147db4d133485bc5d204f888e42b44022f52a710f44c82848bd5e4e.json', 'e59ccdc48b6ac60ec402fe1ed82ba9855e4551d327400e6eadacd4ca.json', 'e6de2462cabaf33f3693cfbf5f39de569d3ff8a07734b65c68fff4d6.json', 'e8149c004cfd63782350f0e16ca68b1733cd03219fc519a0633edc5f.json', 'eb16e46b1a823776c926cb9940ee5e5ad5cf50124356b725d58f3709.json', 'f1731fe1077567f5aab24613a24a146ed077f150762f84da3cdd5ea5.json', 'f8cb8c30e3fb5b6cf32217842330bbdd5a78d48de504a5e3f322b34c.json']
out = out.select(lambda f: f['filename'] not in dropfiles)

print(out)

dfsfds





#############
# accuracy table

# or universal geterror func
def get_gcp_resids(f): 
  # model loo residuals max
  import automap as mapfit
  trans = mapfit.transforms.from_json(f['georef']['transform_estimation']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['georef']['gcps_final']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['georef']['gcps_final']['features']]
  resids = mapfit.accuracy.residuals(trans, inpoints, outpoints, invert=True)
  return resids
def get_loo_resids(f): 
  # model loo residuals max
  import automap as mapfit
  trans = mapfit.transforms.from_json(f['georef']['transform_estimation']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['georef']['gcps_final']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['georef']['gcps_final']['features']]
  resids = mapfit.accuracy.loo_residuals(trans, inpoints, outpoints, invert=True)
  return resids
def getdiag(f): 
  w = f['width']
  h = f['height']
  return math.hypot(w,h)
def getpercerror(f, error_type):
  if error_type == 'loo_max':
    if f['georef'].get('transform_estimation',None):
      resids = get_loo_resids(f)
      err = max(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc
  elif error_type == 'loo_rmse':
    if f['georef'].get('transform_estimation',None):
      resids = get_loo_resids(f)
      import automap as mapfit
      err = mapfit.accuracy.RMSE(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc
    
  elif error_type == 'gcp_max':
    if f['georef'].get('transform_estimation',None):
      resids = get_gcp_resids(f)
      err = max(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc
  elif error_type == 'gcp_rmse':
    if f['georef'].get('transform_estimation',None):
      resids = get_gcp_resids(f)
      import automap as mapfit
      err = mapfit.accuracy.RMSE(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc

import numpy as np
def table(stats, error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        v = getpercerror(f,error_type)
        if f['georef'].get('transform_estimation',None) and math.isnan(v):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['georef'].get('transform_estimation',None):
            return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.copy()
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count',lambda f: 1,'count'),
                                    ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    # first rows
    cumul = 0
    rows = []
    for f in agg:
        cumul += f.row[-1]
        row = f.row + [cumul]
        rows.append(row)
    # then total
    rows.append(['total', len(sub), 100.0, 100.0])
    return rows #np.array(rows)
def tables(stats):
  # concatenate multiple tables horizontally
  rows = table(stats, 'loo_max')
  rows = [[row[0]]+row[2:] for row in rows]
  addrows = table(stats, 'gcp_max')
  for addrow in addrows:
    for row in rows:
      if addrow[0] == row[0]:
        row.extend(addrow[2:])
        break
  
  # format as latex
  for row in rows:
      row[1:] = ['{:.1f}\%'.format(v) for v in row[1:]]
      latexrow = ' & '.join(map(str,row)) + ' \\\\'
      print(latexrow)
      
tables(out)





##############

import matplotlib.pyplot as plt

# histogram of image sizes
ws = [f['width'] for f in out]
plt.hist(ws, bins=50)
avgwidth = sum(ws)/float(len(ws))
print('min width',min(ws))
print('avg width',avgwidth)

# pie chart of georef success
total = len(out)
success = [f for f in out
          if 'transform_estimation' in f['georef']]
success = len(success)
fail = total - success
plt.pie([success,fail],
       labels=['success','fail'],
       autopct='%1.2f%%')

# gcp rmse histogram (note backward is pixels, forward is geodesic)
success = [f for f in out
          if 'transform_estimation' in f['georef']]
def geterror(f): 
  # model error max
  resids = [gcp['properties']['origresidual'] for gcp in f['georef']['gcps_final']['features']]
  err = max(resids)
  return err
def geterror(f): 
  # model loo residuals max
  trans = mapfit.transforms.from_json(f['georef']['transform_estimation']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['georef']['gcps_final']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['georef']['gcps_final']['features']]
  resids = mapfit.accuracy.loo_residuals(trans, inpoints, outpoints, invert=True)
  err = max(resids)
  return err
def geterror2(f):
  # full toponym error max
  backtrans = mapfit.transforms.from_json(f['georef']['transform_estimation']['backward']['model'])
  props = [feat['properties'] for feat in f['georef']['gcps_matched']['features']]
  origpoints = [(f['origx'],f['origy']) for f in props]
  matchpoints = [(f['matchx'],f['matchy']) for f in props]
  matchxs,matchys = zip(*matchpoints)
  predx,predy = backtrans.predict(matchxs, matchys)
  origxs,origys = zip(*origpoints)
  diffs = mapfit.accuracy.distances(origxs, origys, predx, predy)
  err = max(diffs)
  return err

# first pixel error
errors = [geterror(f) for f in success]
plt.hist(errors, bins=50, range=(0,2000)) # pixels
plt.hist(errors, bins=50, range=(0,100)) # pixels

# then percent error
#getdiag = lambda f: math.hypot(f['width'], f['height'])
#getpercerror = lambda f: geterror(f) / float(getdiag(f)/2.0) * 100
errors = [getpercerror(f, 'loo_max') for f in success]
plt.hist(errors, bins=50, range=(0,20)) # percent of image radius
plt.hist(errors, bins=50, range=(0,5)) # percent of image radius








######
success = out

# then counts in accuracy categories (NOTE: Percents is relative to successful ones, ie excludes failed category)
import classypie as cp
for k,group in cp.split(success, key=lambda f: getpercerror(f,'loo_max'), breaks=[0,1,5,20,100,100000000000000]):
    print(k,len(group),len(group)/float(len(out))*100)

# preload global satims
if 1:
    import urllib
    print('dl east')
    urllib.urlretrieve('https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_west.tif',
                       '../data/land_shallow_topo_west.tif'
                      )
    print('dl west')
    urllib.urlretrieve('https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_east.tif',
                       '../data/land_shallow_topo_east.tif'
                      )
    
print('open global sats')
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 466560000*2

scale = 180.0/21600
affine = [scale,0,-180,
         0,-scale,90]
sat_west = pg.RasterData('../data/land_shallow_topo_west.tif', affine=affine)
sat_west.set_geotransform(affine=affine)
affine = [scale,0,0,
         0,-scale,90]
sat_east = pg.RasterData('../data/land_shallow_topo_east.tif', affine=affine)
sat_east.set_geotransform(affine=affine)
def get_sat(bbox, padding=0):
    if padding:
        xmin,ymax,xmax,ymin = bbox
        w,h = abs(xmax-xmin),abs(ymax-ymin)
        xpad,ypad = w*padding, h*padding
        bbox = [xmin-xpad,ymax+ypad,xmax+xpad,ymin-ypad]
    if bbox[0] < 0:
        _bbox = list(bbox)
        _bbox[2] = min(0, _bbox[2])
        yield sat_west.manage.crop(_bbox)
    if bbox[2] >= 0:
        _bbox = list(bbox)
        _bbox[0] = max(0, _bbox[0])
        yield sat_east.manage.crop(_bbox)
  
# test view
from IPython.display import display
def get_url_image(url):
    from PIL import Image
    import urllib2
    import io
    raw = urllib2.urlopen(url).read()
    im = Image.open(io.BytesIO(raw))
    return im
def get_file_image(filename):
    from PIL import Image
    im = Image.open(filename)
    return im
def view_georef(f):
    assert f['width'] < 3000
    fileroot,_ = os.path.splitext(f['filename'])
    ext = '.png' #_,ext = os.path.splitext(f['url'])
    im = get_file_image('scrape/{}{}'.format(fileroot,ext))
    info = mapfit.main.warp_image(im, f['georef']['transform_estimation'])
    pg.vector.data.DEFAULT_SPATIAL_INDEX = 'quadtree'
    
    georef = pg.RasterData(image=info['image'], affine=info['affine'])
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    
    m = pg.renderer.Map(width=3000, height=3000) #,background=(91,181,200))
    
    for sat in get_sat(georef.bbox, padding=0.5):
        for b in sat.bands:
            b.compute('min(val+50, 255)')
        m.add_layer(sat)
        
    #rivers = pg.VectorData('../data/ne_10m_rivers_lake_centerlines.shp')
    #m.add_layer(rivers, fillcolor=(54,115,159,200))
    
    roads = pg.VectorData('../data/ne_10m_roads.shp')
    roads = roads.select(lambda f: not 'Ferry' in f['type'])
    m.add_layer(roads, fillcolor=(0,0,0,255), fillsize='7px')
    m.add_layer(roads, fillcolor=(255,255,0,255), fillsize='5px')
    
    countries = pg.VectorData('../data/ne_10m_admin_0_countries.shp')
    m.add_layer(countries, fillcolor=None, outlinecolor=(255,255,255,200), outlinewidth='5px') #(255,222,173))
    
    m.add_layer(georef, transparency=0.4)
    
    # add toponyms
    topos = pg.VectorData()
    for feat in f['georef']['toponym_candidates']['features']:
        x,y = feat['geometry']['coordinates']
        trans = mapfit.transforms.from_json(f['georef']['transform_estimation']['forward'])
        lon,lat = trans.predict(x, y)
        geom = {'type':'Point', 'coordinates':[lon,lat]}
        topos.add_feature([], geom)
    print(topos)
        
    #m.add_layer(topos, fillcolor=None, outlinecolor='black', outlinewidth=0.3)
    #m.add_layer(topos, fillcolor=None, outlinecolor='red', outlinewidth=0.2)
    
    # add control points
    linedata = pg.VectorData()
    fromdata = pg.VectorData()
    todata = pg.VectorData()
    for feat in f['georef']['gcps_final']['features']:
        props = feat['properties']
        match = props['matchx'],props['matchy']
        match_pred = props['matchx_pred'],props['matchy_pred']
        line = {'type':'LineString',
               'coordinates':[match,match_pred]}
        linedata.add_feature([], line)
        fromdata.add_feature([], {'type':'Point','coordinates':match})
        todata.add_feature([], {'type':'Point','coordinates':match_pred})
    #print(linedata)
    
    #m.add_layer(todata, fillcolor=None, outlinecolor='black', outlinewidth=0.6)
    #m.add_layer(todata, fillcolor=None, outlinecolor=(24+30,128+30,56+30), outlinewidth=0.4)
    m.add_layer(todata, fillcolor=(255,0,0,160), outlinecolor='black', outlinewidth=0.06)
    
    #m.add_layer(linedata, fillcolor='red', fillsize='2px', outlinecolor=None)
    
    m.zoom_bbox(*georef.bbox)
    m.zoom_out(1.1)
    m.render_all()
    return m.img
  
# x maps per accu category
import classypie as cp
for k,group in cp.split(success, key=lambda f: getpercerror(f,'loo_max'), breaks=[0,1,5,20,100,100000000000000]):
    print('=======================================')
    print(k,'accuracy category',len(group),'maps',len(group)/float(len(success))*100,'% of maps')
    sub = [f for f in group 
           if 'transform_estimation' in f['georef'] and f['width'] < 3000]
    for i,f in enumerate(list(sub)[:10]):
        props = f.__geo_interface__['properties']
        print('url',props['url'])
        print('image',props['width'],props['height'])
        print('loo max pixel', geterror(f))
        print('loo_max percent', getpercerror(f,'loo_max'))
        print(props['georef']['error_calculation'])
        #continue
        im = view_georef(f)
        #display(im)
        im.save('analyze/figures/realmaps_mapqual_{}_{}.png'.format(k,i) )

# world map of all georefs? 
import numpy as np
import PIL, PIL.Image
import math

def create_map(feats):
  #m = pg.renderer.Map(width=4000,height=2000,background=(91,181,200))
  #world = pg.VectorData('../data/ne_10m_admin_0_countries.shp',
  #                     )
  #m.add_layer(world, fillcolor=(255,222,173), outlinewidth='2px')
  #m.zoom_bbox(-19,-39,53,42)
  #m.render_all()
  
  pg.vector.data.DEFAULT_SPATIAL_INDEX = 'quadtree'
  m = pg.renderer.Map(4000,2000,background='white',crs='ESRI:54030')

  # water
  w = pg.VectorData()
  top = [(x,90) for x in range(-180,180+1,1)]
  bottom = [(x,-90) for x in reversed(range(-180,180+1,1))]
  left = [(-180,y) for y in range(-90,90+1,1)]
  right = [(180,y) for y in reversed(range(-90,90+1,1))]
  geoj = {'type':'Polygon',
         'coordinates': [top+right+bottom+left]}
  w.add_feature([], geoj)
  m.add_layer(w, fillcolor=(91,181,200))

  # countries
  m.add_layer('../data/ne_10m_admin_0_countries.shp',
             fillcolor=(255,222,173),
             outlinewidth='3px')
  for i,f in enumerate(feats):
    print(f['url'])
    fileroot,_ = os.path.splitext(f['filename'])
    ext = '.png' #_,ext = os.path.splitext(f['url'])
    im = get_file_image('scrape/{}{}'.format(fileroot,ext))
    info = mapfit.main.warp_image(im, f['georef']['transform_estimation'])
    georef = pg.RasterData(image=info['image'], affine=info['affine'])
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    m.add_layer(georef, transparency=0.5)
  print('map count',i)
  return m

# either just for 0-1%
def iterfeats():
  for f in success:
    if 'transform_estimation' in f['georef'] and f['width'] < 3000 and getpercerror(f,'loo_max')<=1:
      yield f
m = create_map(iterfeats())
m.render_all()
m.img
m.img.save('analyze/figures/realmaps_world_0-1.png')

# or for 0-5%
def iterfeats():
  for f in success:
    if 'transform_estimation' in f['georef'] and f['width'] < 3000 and getpercerror(f,'loo_max')<=5:
      yield f
m = create_map(iterfeats())
m.render_all()
m.img
m.img.save('analyze/figures/realmaps_world_0-5.png')

# or for 0-20%
def iterfeats():
  for f in success:
    if 'transform_estimation' in f['georef'] and f['width'] < 3000 and getpercerror(f,'loo_max')<=20:
      yield f
m = create_map(iterfeats())
m.render_all()
m.img
m.img.save('analyze/figures/realmaps_world_0-20.png')

# or for 0-100%
def iterfeats():
  for f in success:
    if 'transform_estimation' in f['georef'] and f['width'] < 3000 and getpercerror(f,'loo_max')<=100:
      yield f
m = create_map(iterfeats())
m.render_all()
m.img
m.img.save('analyze/figures/realmaps_world_0-100.png')

    
    
