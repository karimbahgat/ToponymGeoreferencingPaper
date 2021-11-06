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
    os.chdir('simulations')
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

print out

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
                       'data/land_shallow_topo_west.tif'
                      )
    print('dl west')
    urllib.urlretrieve('https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_east.tif',
                       'data/land_shallow_topo_east.tif'
                      )
    
print('open global sats')
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 466560000*2

scale = 180.0/21600
affine = [scale,0,-180,
         0,-scale,90]
sat_west = pg.RasterData('data/land_shallow_topo_west.tif', affine=affine)
sat_west.set_geotransform(affine=affine)
affine = [scale,0,0,
         0,-scale,90]
sat_east = pg.RasterData('data/land_shallow_topo_east.tif', affine=affine)
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
        
    #rivers = pg.VectorData('data/ne_10m_rivers_lake_centerlines.shp')
    #m.add_layer(rivers, fillcolor=(54,115,159,200))
    
    roads = pg.VectorData('data/ne_10m_roads.shp')
    roads = roads.select(lambda f: not 'Ferry' in f['type'])
    m.add_layer(roads, fillcolor=(0,0,0,255), fillsize='7px')
    m.add_layer(roads, fillcolor=(255,255,0,255), fillsize='5px')
    
    countries = pg.VectorData('data/ne_10m_admin_0_countries.shp')
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
  #world = pg.VectorData('data/ne_10m_admin_0_countries.shp',
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
  m.add_layer('data/ne_10m_admin_0_countries.shp',
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
    
    
    
    
    
    
    
    
    
    
    
    



fdsafas




db = sqlite3.connect('analyze/stats.db')
stats = pg.VectorData()
stats.fields = ['sim_id', 'rendopts', 'controlpoints', 'matched', 'segmentation', 'text', 'toponyms', 'transform', 'accuracy', 'timings', 'log']
for row in db.execute('select * from data'):
    row = list(row)
    stats.add_feature(row, None)

# load json
for fl in stats.fields[1:]: # skip sim_id (not json)
    stats.compute(fl, lambda f: json.loads(f[fl]))
stats.compute('log', lambda f: f['log']['text'] if f['log'] else None)

# all renderings
stats_all = stats
print(stats_all)

# begin filtering down
stats = stats_all

# for some reason, log.txt files were also included, exclude these
stats = stats.select(lambda f: not f['sim_id'].endswith('log.txt'))

# exclude maps with nonlegible texts (jpg at 1000 pixel resolution)
stats = stats.select(lambda f: not f['sim_id'].endswith('_6'))
print(stats)

# scenes
scenes = set([f['sim_id'].split('_')[1] for f in stats])
print(len(scenes))

# maps per scene
for scene in sorted(scenes):
    count = len(stats.select(lambda f: f['sim_id'].split('_')[1]==scene))
    print(scene, count)

fdafa







# misc exploration
import math
print(stats)
print(stats.select(lambda f: f['accuracy']))
print(stats.select(lambda f: f['accuracy'] and not math.isnan(f['accuracy']['max_georeferenced']['percent'])))
isnan = stats.select(lambda f: f['accuracy'] and math.isnan(f['accuracy']['max_georeferenced']['percent']))

for f in stats.select(lambda f: f['accuracy']).aggregate(['sim_id'], fieldmapping=[('max',lambda f: f['accuracy']['max_georeferenced']['percent'],'max')]):
    print(f.row)
  
for f in stats.select(lambda f: f['sim_id']=='sim_75_12_5'):
    print(f['accuracy']['max_georeferenced']['percent'])
  
import math
acc = [f['accuracy']['max_georeferenced']['percent']
       for f in stats.select(lambda f: f['accuracy'] 
                             and f['accuracy']['max_georeferenced']['percent']
                             and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                            )]
plt.hist(acc, range=(0,1))
  
stats.select(lambda f: f['accuracy'] 
                       and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                      and f['accuracy']['max_georeferenced']['percent'] < 0.2
                      )
    
  
    
    
    

    
### JOURNAL ARTICLE
from IPython.display import display
import numpy as np

sns.set_style("darkgrid")

# km error histograms, by extent
def plot(error_type='rmse', range=None, bins=10):
    fig,axes = plt.subplots(1, 4, 
                            sharex=True, sharey=True,
                            figsize=(8,2))
    
    # 50deg ie 5000km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==50)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[0]
    ax.set_title('5000 km extent')
    #ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 10deg ie 1000km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==10)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[1]
    ax.set_title('1000 km extent')
    #ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 1deg ie 100km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==1)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[2]
    ax.set_title('100 km extent')
    #ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 0.25deg ie 25km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==0.25)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[3]
    ax.set_title('25 km extent')
    #ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # save
    fig.subplots_adjust(bottom=0.1)
    fig.savefig('analyze/figures/error_km_by_extent.png')
    
plot('max', range=(0,5000))

# perc error histograms, max error vs rmse error
def plot(range=None, bins=10):
    fig,axes = plt.subplots(1, 2, 
                            sharex=True, sharey=True,
                            figsize=(8,4))
    
    # max
    sub = stats.select(lambda f: f['accuracy'])
    errors = [f['accuracy']['max_georeferenced']['percent']*100 for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[0]
    ax.set_title('Max error')
    ax.set_yscale('log')
    ax.set_xlabel('error as % of image radius')
    ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # rmse
    sub = stats.select(lambda f: f['accuracy'])
    errors = [f['accuracy']['rmse_georeferenced']['percent']*100 for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[1]
    ax.set_title('RMSE error')
    ax.set_yscale('log')
    ax.set_xlabel('error as % of image radius')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # save
    plt.savefig('analyze/figures/error_perc_max_vs_rmse.png')
    
plot(range=(0,200))


# ABOVE NOT USED









# map param stats

# define params
resolutions = [3000, 2000, 1000]
extents = [50, 10, 1, 0.25] # ca 5000km, 1000km, 100km, and 10km
quantities = [80, 40, 20, 10]
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
distributions = ['random'] 
projections = ['eqc',
                'lcc', 
               'tmerc',
               ] 
datas = [False, 
         True] 
metas = [False, # nothing
         True, # text noise + meta boxes (arealabels + title + legend) + gridlines
         ]
imformats = ['png','jpg']
paramvals = dict(extents=extents,
              toponyms=quantities,
              distributions=distributions,
              uncertainties=uncertainties,
              projections=projections,
              resolutions=resolutions,
              imformats=imformats,
              metas=metas,
              datas=datas,
                )
paramkeys = dict(extents=lambda f: f['rendopts']['regionopts']['extent'],
              toponyms=lambda f: f['rendopts']['placeopts']['quantity'],
              distributions=lambda f: f['rendopts']['placeopts']['distribution'],
              uncertainties=lambda f: f['rendopts']['placeopts']['uncertainty'],
              projections=lambda f: f['rendopts']['projection'].split()[0].replace('+proj=',''),
              resolutions=lambda f: f['rendopts']['noiseopts']['resolution'],
              imformats=lambda f: f['rendopts']['noiseopts']['format'],
              datas=lambda f: len(f['rendopts']['datas'])>1,
              metas=lambda f: f['rendopts']['metaopts']['arealabels'])
paramlabels = dict(extents='mapExtent (km)',
              toponyms='numToponyms',
              distributions='toponymDispersed',
              uncertainties='toponymUncertainty (km)',
              projections='mapProjection',
              resolutions='imgResolution',
              imformats='pixelNoise',
              metas='metaNoise',
              datas='dataNoise',
                )

for k,func in paramkeys.items():
    statscopy = stats.select(lambda f: f['rendopts'])
    statscopy.compute('group', func)
    label = paramlabels[k]
    print(label)
    agg = statscopy.aggregate(['group'], fieldmapping=[('count','sim_id','count')])
    for f in agg:
        row = ['',''] + map(str, f.row)
        print(' & '.join(row) + ' \\\\')









# example map error, error surface vs map overlay
def inspect_georef_errors(georef_fil, truth_fil, error_type):
    # georef errors
    mapp = mapfit.debug.render_georeferencing_errors(georef_fil, truth_fil, error_type)
    mapp.render_all()
    return mapp.img
def debug_sim_id(sim_id):
    if sim_id.endswith(('2','4','6')):
        ext = 'jpg'
    else:
        ext = 'png'
    fil = 'maps/{}_image.{}'.format(sim_id,ext)
    outfil = 'output/{}_georeferenced_auto.tif'.format(sim_id)
    img = inspect_georef_errors(outfil, fil, 'pixel')
    print('img size',img.size)
    return img
def map_overlay(sim_id):
    if sim_id.endswith(('2','4','6')):
        ext = 'jpg'
    else:
        ext = 'png'
    fil = 'maps/{}_image.{}'.format(sim_id,ext)
    outfil = 'output/{}_georeferenced_auto.tif'.format(sim_id)
    import pythongis as pg
    truth = pg.RasterData(fil)
    print(truth.crs)
    m = pg.renderer.Map(800,800,crs=truth.crs)
    m.add_layer(truth, transparency=0.5)
    georef = pg.RasterData(outfil)
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    print(georef.crs)
    m.add_layer(georef, transparency=0.5)
    m.zoom_auto()
    m.render_all()
    return m.img
def plot(sim_id, file_suffix):
    fig,axes = plt.subplots(1, 2, 
                            figsize=(8,3))
    # error surface
    im = debug_sim_id(sim_id)
    ax = axes[0]
    ax.set_title('a)')
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(im)
    
    # map overlay
    im = map_overlay(sim_id)
    ax = axes[1]
    ax.set_title('b)')
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(im)
    
    # save
    plt.savefig('analyze/figures/errors_maps_{}.png'.format(file_suffix))
def subplot(ax, sim_id, subtitle):
    # map overlay
    im = map_overlay(sim_id)
    ax.set_title(subtitle)
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(im)
    
# ??? 
#examps = [f for f in stats if f['accuracy'] and 800 < f['accuracy']['rmse_georeferenced']['geographic'] < 1200]
#for f in examps[0:1]:
#    print f['accuracy']['rmse_georeferenced']
#    display(map_overlay(f['sim_id']))

# fig
fig,axes = plt.subplots(3, 2, 
                        figsize=(6,10))
# out of wack
examps = [f for f in stats if f['accuracy'] and 2 < f['accuracy']['max_georeferenced']['percent'] < 3]
for f in examps[8:9]:
    print(f['accuracy']['max_georeferenced'])
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_wack.png')
    #plot(f['sim_id'], 'wack')
    #subplot(axes[2,0], f['sim_id'], 'not usable\n'+'>100%')
# approx but unusable
examps = [f for f in stats if f['accuracy'] and 0.5 < f['accuracy']['max_georeferenced']['percent'] < 0.6]
for f in examps[2:3]:
    print(f['accuracy']['max_georeferenced'])
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_needsadj.png')
    #plot(f['sim_id'], 'approx_bad')
    #subplot(axes[1,1], f['sim_id'], 'approximate, needs fixing\n'+'>20%')
# approx
examps = [f for f in stats if f['accuracy'] and 0.1 < f['accuracy']['max_georeferenced']['percent'] < 0.12]
for f in examps[:1]:
    print(f['accuracy']['max_georeferenced'])
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_approx.png')
    #plot(f['sim_id'], 'approx')
    #subplot(axes[1,0], f['sim_id'], 'approximate\n'+'>5%')
# reasonable
examps = [f for f in stats if f['accuracy'] and 0.01 < f['accuracy']['max_georeferenced']['percent'] < 0.05]
for f in examps[:1]:
    print(f['accuracy']['max_georeferenced'])
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_reasonable.png')
    #plot(f['sim_id'], 'reasonable')
    #subplot(axes[0,1], f['sim_id'], 'reasonable\n'+'>1%')
# excellent
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] < 0.01]
for f in examps[:1]:
    print(f['accuracy']['max_georeferenced'])
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_excellent.png')
    #plot(f['sim_id'], 'excellent')
    #subplot(axes[0,0], f['sim_id'], 'excellent\n'+'<1%')

fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95,
                wspace=0.05) #, hspace=0.05)
fig.delaxes(axes[2,1])
fig.savefig('analyze/figures/errors_maps_categories.png')

# table of overall simulation accuracy
def table(error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        if f['accuracy'] and math.isnan(f['accuracy'][error_type+'_georeferenced']['percent']):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            v = f['accuracy'][error_type+'_georeferenced']['percent']
            return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0] if not math.isnan(v) else None
        else:
            # no accuracy dict (failed)
            return 6
    sub = stats.copy()
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
                                    ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    cumul = 0
    for f in agg:
        cumul += f.row[-1]
        row = f.row + [cumul]
        row[-2:] = [round(v,1) for v in row[-2:]]
        latexrow = ' & '.join(map(str,row)) + ' \\\\'
        print latexrow
        
    #print ['failed', len(stats)-len(success), (len(stats)-len(success))/float(len(stats))*100, 'NA']
    row = ['total', len(stats), 100.0, 100.0]
    latexrow = ' & '.join(map(str,row)) + ' \\\\'
    print latexrow

table('max')

# table of timings
# cols: total,avg,median
# rows: each processing step
# note: total of 15 days, but down to ca 2 days bc of multiple cores
def table():
    sub = stats.select(lambda f: f['timings'])
    sub.compute('timings', lambda f: f['timings'].copy())
    rows = []
    for f in sub:
        f['timings']['total_nowarp'] = f['timings']['total'] - f['timings']['warping']
    for k in 'segmentation text_recognition toponym_candidates gcps_matched transform_estimation total_nowarp'.split():
        sub.compute(k, lambda f: f['timings'][k])
        #print min((f[k] for f in sub))
        #print max((f[k] for f in sub))
        agg = sub.aggregate(key=lambda f: None,
                            fieldmapping=[('avg',k,'avg'),
                                          ('median',k,lambda vals: sorted(vals)[len(vals)//2]),
                                          ('total',k,'sum')
                                          ]
                            )
        f = agg[1]
        row = [k] + [v for v in f.row]
        rows.append(row)
    
    avgtot = rows[-1][1] #sum((r[1] for r in rows[:-1]))
    medtot = rows[-1][2] #sum((r[2] for r in rows[:-1]))
    totaltot = rows[-1][3] #sum((r[3] for r in rows[:-1]))
    for row in rows:
        perc = row[1] / float(avgtot) * 100
        row[1] = '{:.1f}s ({:.1f}\%)'.format(row[1], perc)
        
        perc = row[2] / float(medtot) * 100
        row[2] = '{:.1f}s ({:.1f}\%)'.format(row[2], perc)
        
        perc = row[3] / float(totaltot) * 100
        row[3] = '{:.1f}h ({:.1f}\%)'.format(row[3]/60.0/60.0, perc)
        
        #print row
        latexrow = ' & '.join(map(str,row)) + ' \\\\'
        print latexrow

table()

# table of accuracy for realistic quality maps
def table():
    # x different centers
    import classypie as cp
    def classfunc(f): 
        if f['accuracy'] and math.isnan(f['accuracy']['max_georeferenced']['percent']):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            v = f['accuracy']['max_georeferenced']['percent']
            return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.copy()
    sub = sub.select(lambda f: f['rendopts'] and \
                   f['rendopts']['noiseopts']['resolution']>1500 and \
                   #f['rendopts']['noiseopts']['format']=='png' and \
                   f['rendopts']['placeopts']['quantity']>=40 and \
                   f['rendopts']['placeopts']['uncertainty']<=0.1 and \
                   #f['rendopts']['placeopts']['distribution']=='random' and \
                   #f['rendopts']['projection']==None and \
                   #f['rendopts']['metaopts']['arealabels']==False
                   f['rendopts']['regionopts']['extent']>=1
                  )
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
                                    ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    cumul = 0
    for f in agg:
        cumul += f.row[-1]
        row = f.row + [cumul]
        row[-2:] = ['{:.1f}\%'.format(v) for v in row[-2:]]
        latexrow = ' & '.join(map(str,row)) + ' \\\\'
        print latexrow
        
    #print ['failed', len(stats)-len(success), (len(stats)-len(success))/float(len(stats))*100, 'NA']
    row = ['total', len(sub), 100.0, 100.0]
    latexrow = ' & '.join(map(str,row)) + ' \\\\'
    print latexrow

table()



# boxplots
# define params
resolutions = [3000, 2000, 1000]
extents = [50, 10, 1, 0.25] # ca 5000km, 1000km, 100km, and 10km
quantities = [80, 40, 20, 10]
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
distributions = ['random'] 
projections = ['eqc',
                'lcc', 
               'tmerc',
               ] 
datas = [False, 
         True] 
metas = [False, # nothing
         True, # text noise + meta boxes (arealabels + title + legend) + gridlines
         ]
imformats = ['png','jpg']
paramvals = dict(extents=extents,
              toponyms=quantities,
              distributions=distributions,
              uncertainties=uncertainties,
              projections=projections,
              resolutions=resolutions,
              imformats=imformats,
              metas=metas,
              datas=datas,
                )
paramkeys = dict(extents=lambda f: f['rendopts']['regionopts']['extent'],
              toponyms=lambda f: f['rendopts']['placeopts']['quantity'],
              distributions=lambda f: f['rendopts']['placeopts']['distribution'],
              uncertainties=lambda f: f['rendopts']['placeopts']['uncertainty'],
              projections=lambda f: f['rendopts']['projection'].split()[0].replace('+proj=',''),
              resolutions=lambda f: f['rendopts']['noiseopts']['resolution'],
              imformats=lambda f: f['rendopts']['noiseopts']['format'],
              datas=lambda f: len(f['rendopts']['datas'])>1,
              metas=lambda f: f['rendopts']['metaopts']['arealabels'])
paramlabels = dict(extents='mapExtent (km)',
              toponyms='numToponyms',
              distributions='toponymDispersed',
              uncertainties='toponymUncertainty (km)',
              projections='mapProjection',
              resolutions='imgResolution',
              imformats='pixelNoise',
              metas='metaNoise',
              datas='dataNoise',
                )
  
# boxplots for each param value
def printstats(errors):
    import numpy as np
    errors = np.array(errors)
    print '-> count', len(errors)
    if not len(errors):
        return
    print '-> median', np.median(errors)
    print '-> mean', errors.mean()
    print '-> max', errors.max()

def boxplot_stats(ax, param, vals, xlabel=True, **baselinevals):
    print param
    #ax.set_title(param)
    if xlabel:
        paramlabel = paramlabels[param]
        ax.set_xlabel(paramlabel, weight='bold')
    if axes.ndim == 1 and ax in (axes[0],axes[3],axes[6]):
        ax.set_ylabel('max error (%)')
    valerrors = []
    def is_param_baseline(f):
        # NO LONGER SETS ALL TO BASELINE, ONLY CUSTOM BASELINE VALS
        return all([paramkeys[k](f)==v for k,v in baselinevals.items()])
        # IGNORES BELOW
        checks = []
        for par,parvals in paramvals.items():
            if par == param: 
                continue
            val = paramkeys[par](f)
            baselineval = baselinevals.get(par, parvals[0])
            checks.append(val==baselineval)
        # select only if all param values are at baseline (pos 0)
        return all(checks)
    for val in vals:
        sub = stats.select(lambda f: f['rendopts'] and f['accuracy'] and is_param_baseline(f) and paramkeys[param](f) == val)
        errors = [f['accuracy']['max_georeferenced']['percent']*100 for f in sub]
        errors = [e for e in errors if not math.isnan(e)]
        print 'stats for {}={} (n={})'.format(param, val, len(errors))
        printstats(errors)
        valerrors.append(errors)
    if param in 'extents uncertainties':
        vals = [int(v*100) for v in vals]
    _ = ax.boxplot(valerrors, labels=[str(v)[:12] for v in vals],
                   showcaps=False, showfliers=False, vert=True)

fig,axes = plt.subplots(3, 3, 
                        #sharex=True, sharey=True,
                        figsize=(8,8))
axes = axes.flatten()
    
boxplot_stats(axes[0], 'extents', extents, )

boxplot_stats(axes[1], 'resolutions', resolutions, )

boxplot_stats(axes[2], 'projections', projections, )

boxplot_stats(axes[3], 'datas', datas, )

boxplot_stats(axes[4], 'metas', metas, )

boxplot_stats(axes[5], 'imformats', imformats, )

boxplot_stats(axes[6], 'toponyms', quantities, )

boxplot_stats(axes[7], 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05,
                   hspace=0.30) #wspace=0.05)
fig.delaxes(axes[-1])
#fig.suptitle('Effect of each parameter \n while keeping all others at baseline')
fig.savefig('analyze/figures/errors_baseline_boxplots.png')



# same boxplots but broken down by extent
fig,axes = plt.subplots(7, 4, 
                        #sharex=True, sharey=True,
                        figsize=(8,10))

j = 0

ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('img\nResolution', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    axes[0,i].set_title('extent = {} km'.format(int(extents[i]*100)))
    boxplot_stats(axes[0,i], 'resolutions', resolutions, xlabel=False, extents=extents[i])

j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('map\nProjection', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'projections', projections, xlabel=False, extents=extents[i])

j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('data\nNoise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'datas', datas, xlabel=False, extents=extents[i])
    
j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('meta\nNoise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'metas', metas, xlabel=False, extents=extents[i])

j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('pixel\nNoise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'imformats', imformats, xlabel=False, extents=extents[i])
    
j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('num\nToponyms', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'toponyms', quantities, xlabel=False, extents=extents[i])

j += 1
ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('toponym\nUncertainty (km)', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'uncertainties', uncertainties, xlabel=False, extents=extents[i])

fig.subplots_adjust(left=0.1, bottom=0.04, right=0.99, top=0.96,
                   hspace=0.4, wspace=0.4)
#fig.suptitle('Effect of each parameter compared to baseline, by map extent')
fig.savefig('analyze/figures/errors_baseline_boxplots_by_extent.png')

    
  
# same but bargraphs of success rate (<100)

import numpy as np

# bargraphs for each param value
def outcomeplot_stats(ax, param, vals, xlabel=True, **baselinevals):
    print param
    #ax.set_title(param)
    if xlabel:
        paramlabel = paramlabels[param]
        ax.set_xlabel(paramlabel, weight='bold')
    if axes.ndim == 1 and ax in (axes[0],axes[3],axes[6]):
        ax.set_ylabel('% of maps')
    rates1 = []
    rates2 = []
    #rates3 = []
    def is_param_baseline(f):
        # NO LONGER SETS ALL TO BASELINE, ONLY CUSTOM BASELINE VALS
        return all([paramkeys[k](f)==v for k,v in baselinevals.items()])
        # IGNORES BELOW
        checks = []
        for par,parvals in paramvals.items():
            if par == param: 
                continue
            val = paramkeys[par](f)
            baselineval = baselinevals.get(par, parvals[0])
            checks.append(val==baselineval)
        # select only if all param values are at baseline (pos 0)
        return all(checks)
    for val in vals:
        sub = stats.select(lambda f: f['rendopts'] and is_param_baseline(f) and paramkeys[param](f) == val)
        outcomes = np.array([f['accuracy']['max_georeferenced']['percent'] if f['accuracy'] and not math.isnan(f['accuracy']['max_georeferenced']['percent']) else 999999999999999
                            for f in sub])
        #rate1 = (outcomes!=999999999999999).mean() * 100 # not fail
        rate1 = (outcomes<1.0).mean() * 100 # usable
        #rate2 = (outcomes<0.2).mean() * 100 # no fixing
        rate2 = (outcomes<0.05).mean() * 100 # human-equivalent
        print 'stats for {}={} (n={})'.format(param, val, len(outcomes))
        print rate1,rate2 #,rate3
        rates1.append(rate1)
        rates2.append(rate2)
        #rates3.append(rate3)
    labelvals = list(vals)
    if param in 'extents uncertainties':
        labelvals = [int(v*100) for v in vals]
    w = 0.35
    _ = ax.bar([x for x in range(len(rates1))], rates1, 
               #tick_label=[str(v)[:12] for v in labelvals], 
               align='edge',
               color='tab:blue', width=w, 
               label='Usable (<100% error)',
                )
    _ = ax.bar([x+w for x in range(len(rates2))], rates2, 
               tick_label=[str(v)[:12] for v in labelvals], 
               align='edge',
               color='tab:orange', width=w, 
               label='Human equivalent (<5% error)',
                )
    #_ = ax.bar([x+w+w for x in range(len(rates3))], rates3, 
    #           #tick_label=[str(v)[:12] for v in labelvals], align='edge',
    #           color='tab:green', width=w, 
    #            )

fig,axes = plt.subplots(3, 3, 
                        sharey=True, #sharex=True, sharey=True,
                        figsize=(8,6))
axes = axes.flatten()
    
outcomeplot_stats(axes[0], 'extents', extents, )

outcomeplot_stats(axes[1], 'resolutions', resolutions, )

outcomeplot_stats(axes[2], 'projections', projections, )

outcomeplot_stats(axes[3], 'datas', datas, )

outcomeplot_stats(axes[4], 'metas', metas, )

outcomeplot_stats(axes[5], 'imformats', imformats, )

outcomeplot_stats(axes[6], 'toponyms', quantities, )

outcomeplot_stats(axes[7], 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15,
                   hspace=0.4) #wspace=0.05)
fig.delaxes(axes[-1])
#fig.suptitle('Effect of each parameter \n while keeping all others at baseline')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False,
          loc='lower center')
fig.savefig('analyze/figures/outcomes_baseline_barplots.png')
  
  
###
# same barplots but broken down by extent
fig,axes = plt.subplots(3, 4, 
                        sharey=True, #sharex=True, sharey=True,
                        figsize=(8,6))

j = 0
ax = axes[j,0]
ax.set_ylabel('% of maps')
ax.annotate('map\nProjection', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    axes[0,i].set_title('extent = {} km'.format(extents[i]*100))
    outcomeplot_stats(axes[j,i], 'projections', projections, xlabel=False, extents=extents[i])

j += 1
ax = axes[j,0]
ax.set_ylabel('% of maps')
ax.annotate('data\nNoise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    outcomeplot_stats(axes[j,i], 'datas', datas, xlabel=False, extents=extents[i])

#j += 1
#ax = axes[j,0]
#ax.set_ylabel('% of maps')
#ax.annotate('num\nToponyms', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
#                  xycoords=ax.yaxis.label, textcoords='offset points',
#                  ha='center', va='center', rotation='vertical')
for i in range(4):
    outcomeplot_stats(axes[j,i], 'toponyms', quantities, xlabel=False, extents=extents[i])
    
j += 1
ax = axes[j,0]
ax.set_ylabel('% of maps')
ax.annotate('toponym\nUncertainty (km)', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    outcomeplot_stats(axes[j,i], 'uncertainties', uncertainties, xlabel=False, extents=extents[i])

fig.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.92,
                   hspace=0.4, wspace=0.4)
#fig.suptitle('Effect of each parameter compared to baseline, by map extent')
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False,
          loc='lower center')
fig.savefig('analyze/figures/outcomes_baseline_barplots_by_extent.png')


    
    
# table of error sources (NOT USING, REMOVE?)
def table():
    import classypie as cp
    def classfunc(f): 
        if f['accuracy']:
            v = f['accuracy']['max_georeferenced']['percent']
            return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.select(lambda f: f['rendopts'])
    sub.compute('errclass', classfunc)
    print sub
    agg = sub.aggregate(key=['errclass'],
                    fieldmapping=[('count',lambda f: 1,'count'),
                                  ('extent',lambda f: f['rendopts']['regionopts']['extent']*100,'avg'),
                                  ('resolution',lambda f: f['rendopts']['resolution'],'avg'),
                                  ('quantity',lambda f: f['rendopts']['placeopts']['quantity'],'avg'),
                                  ('uncertainty',lambda f: f['rendopts']['placeopts']['uncertainty']*100,'avg'),
                                  ('gcps',lambda f: f['accuracy']['labels'] * f['accuracy']['labels_used'] if f['accuracy'] else None,'avg'),
                                  #distribution
                                  ('dispersed', lambda f: 1 if f['rendopts']['placeopts']['distribution'] == 'dispersed' else 0, 'avg'),
                                  #projection
                                  ('projection', lambda f: 1 if f['rendopts']['projection'] else 0, 'avg'),
                                  #data noise
                                  ('datanoise', lambda f: 1 if f['rendopts']['datas'] else 0, 'avg'),
                                  #text noise
                                  ('metanoise', lambda f: 1 if f['rendopts']['metaopts']['legend'] else 0, 'avg'),
                                  #pixel noise
                                  ('formatnoise', lambda f: 1 if f['rendopts']['noiseopts']['format']=='jpg' else 0, 'avg'),
                                 ]
                   )
    for f in agg:
        yield f
         
for f in []: #table():
    latexrow = '{errclass} & {extent:.0f} & {resolution:.0f} & {quantity:.1f} & {uncertainty:.2f} & {dispersed:.2f} & {gcps:.1f} & {projection:.2f} & {datanoise:.2f} & {metanoise:.2f} & {formatnoise:.2f} \\\\'.format(errclass=f['errclass'],
                                                                                                                                  extent=f['extent'],
                                                                                                                                  resolution=f['resolution'],
                                                                                                                                  quantity=f['quantity'],
                                                                                                                                  uncertainty=f['uncertainty'],
                                                                                                                                  gcps=f['gcps'] or float('nan'),
                                                                                                                                  dispersed=f['dispersed'],
                                                                                                                                  projection=f['projection'],
                                                                                                                                  datanoise=f['datanoise'],
                                                                                                                                  metanoise=f['metanoise'],
                                                                                                                                  formatnoise=f['formatnoise'],
                                                                                                                                  )
    print latexrow
    
aggrows = []
for f in table():
    if not f['errclass']:
        continue
    row = [f['errclass'],
          format(f['extent'], '.0f'),
          format(f['resolution'], '.0f'),
          format(f['quantity'], '.1f'),
          format(f['uncertainty'], '.2f'),
           format(f['dispersed'], '.2f'),
          format(f['gcps'] or float('nan'), '.2f'),
          format(f['projection'], '.2f'),
          format(f['datanoise'], '.2f'),
          format(f['metanoise'], '.2f'),
          format(f['formatnoise'], '.2f')]
    aggrows.append(row)
errlabels = ['',
             'mapExtent (km)',
             'imgResolution',
            'numToponyms',
             'toponymUncertainty (km)',
            'toponymDispersed',
             'numControlPoints',
            'mapProjection',
            'dataNoise',
            'metaNoise',
             'pixelNoise',
            ]
import numpy as np
for i,row in enumerate(np.transpose(aggrows)):
    row = [errlabels[i]] + list(row)
    print ' & '.join(row) + ' \\\\'
    
# table of linear regression results
import numpy as np
import statsmodels.api as sm

x = np.array([[f['rendopts']['regionopts']['extent'] if f['rendopts'] else float('nan'),
               f['rendopts']['resolution'] if f['rendopts'] else float('nan'),
               f['rendopts']['placeopts']['quantity'] if f['rendopts'] else float('nan'),
               f['rendopts']['placeopts']['uncertainty'] if f['rendopts'] else float('nan'),
                #gcps
                f['accuracy']['labels'] * f['accuracy']['labels_used'] if f['accuracy'] else float('nan'),
                #projection
                (1 if f['rendopts']['projection'] else 0) if f['rendopts'] else float('nan'),
                #data noise
                (1 if f['rendopts']['datas'] else 0) if f['rendopts'] else float('nan'),
                #text noise
                (1 if f['rendopts']['metaopts']['legend'] else 0) if f['rendopts'] else float('nan'),
                #pixel noise
                (1 if f['rendopts']['noiseopts']['format']=='jpg' else 0) if f['rendopts'] else float('nan'),
              ]
              for f in stats])
xnames = ['Constant', 'mapExtent', 'imgResolution', 'numToponyms', 'toponymUncertainty', 'numControlPoints', 'mapProjection', 'dataNoise', 'metaNoise', 'pixelNoise']
y = np.array([f['accuracy']['max_georeferenced']['percent'] if f['accuracy'] else float('nan')
              for f in stats])
ynames = ['Max Error (%)']

x = sm.add_constant(x)
results = sm.OLS(y, x, missing='drop').fit()
print results.summary(yname=ynames, xname=xnames).as_latex()


# table of logistic regression results
import numpy as np
import statsmodels.api as sm

x = np.array([[f['rendopts']['regionopts']['extent'] if f['rendopts'] else float('nan'),
               f['rendopts']['resolution'] if f['rendopts'] else float('nan'),
               f['rendopts']['placeopts']['quantity'] if f['rendopts'] else float('nan'),
               f['rendopts']['placeopts']['uncertainty'] if f['rendopts'] else float('nan'),
                #projection eqc
                1 if 'eqc' in f['rendopts']['projection'] else 0,
                #projection lcc
                1 if 'lcc' in f['rendopts']['projection'] else 0,
                #projection tmerc
                1 if 'tmerc' in f['rendopts']['projection'] else 0,
                #data noise
                (1 if f['rendopts']['datas'] else 0) if f['rendopts'] else float('nan'),
                #text noise
                (1 if f['rendopts']['metaopts']['legend'] else 0) if f['rendopts'] else float('nan'),
                #pixel noise
                (1 if f['rendopts']['noiseopts']['format']=='jpg' else 0) if f['rendopts'] else float('nan'),
              ]
              for f in stats])
for i in range(x.shape[1]):
    # min-max normalize predictors
    v = x[:,i][~np.isnan(x[:,i])]
    x[:,i] = (x[:,i] - v.min()) / (v.max() - v.min())
xnames = ['Constant', 'mapExtent', 'imgResolution', 'numToponyms', 'toponymUncertainty', 'mapProjectionEqc', 'mapProjectionLcc', 'mapProjectionTmerc', 'dataNoise', 'metaNoise', 'pixelNoise']
y = np.array([f['accuracy']['max_georeferenced']['percent'] if f['accuracy'] else 999999999999999
              for f in stats])
y[y<0.05] = 1
y[(0.05<y) & (y<1)] = float('nan')
y[y>1] = 0
ynames = 'Success'

x = sm.add_constant(x)
results = sm.Logit(y, x, missing='drop').fit()
print results.summary(yname=ynames, xname=xnames).as_latex()
    
    
    
