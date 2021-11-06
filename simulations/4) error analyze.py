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
def collect(proctype):
    print 'collecting', proctype
    out = pg.VectorData()
    out.fields = ['sim_id', 'rendopts', 'controlpoints', 'matched', 'segmentation', 'text', 'toponyms', 'transform', 'controlpoints_accuracy', 'accuracy', 'timings', 'log']

    for i,root in enumerate(roots):
        vals = {'sim_id':root}

        path = 'maps/{}_opts.json'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
        
                if not (dat['datas'] and dat['metaopts']['legend']):
                    # must have datalayers and metadata
                    continue
                
                vals['rendopts'] = dat
                
        print(proctype, i, root)
                
        path = 'output/{}_{}_controlpoints.geojson'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['controlpoints'] = dat
                
        path = 'output/{}_{}_debug_gcps_matched.geojson'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['matched'] = dat
                
        path = 'output/{}_{}_debug_segmentation.geojson'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['segmentation'] = dat
                
        path = 'output/{}_{}_debug_text.geojson'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['text'] = dat
                
        path = 'output/{}_{}_debug_text_toponyms.geojson'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['toponyms'] = dat
                
        path = 'output/{}_{}_transform.json'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['transform'] = dat

        path = 'output/{}_{}_errors.json'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['controlpoints_accuracy'] = dat
                
        path = 'output/{}_{}_simulation_errors.json'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['accuracy'] = dat  # true simulation error
                
        path = 'output/{}_{}_debug_timings.json'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['timings'] = dat
                
        path = 'output/{}_{}_log.txt'.format(root, proctype)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = fobj.read()
                vals['log'] = {'text':dat}

        #print vals
        
        out.add_feature(vals, None)
        
    return out
  
def insert_into_db(db, table, data):
    print 'inserting into db', table
    cur = db.cursor()
    
    # encode dicts to json for saving
    print 'json dumping'
    for fl in data.fields[1:]: # skip sim_id (not json)
        data.compute(fl, lambda f: json.dumps(f[fl]))
        
    # create table
    fielddef = ','.join(['{} text'.format(fl) for fl in data.fields])
    cur.execute('create table {} ({})'.format(table, fielddef))
    
    # insert
    print 'inserting'
    for f in data:
        qs = ','.join('?'*len(data.fields))
        vals = tuple(f.row)
        cur.execute('insert into {} values ({})'.format(table, qs), vals)
    db.commit()

if False:
    # setup db
    print 'db init'
    assert not os.path.lexists('analyze/stats_baseline.db')
    ##os.remove('analyze/stats_baseline.db')
    db = sqlite3.connect('analyze/stats_baseline.db')
    roots = set(('_'.join(fil.split('_')[:4]) for fil in os.listdir('maps') if fil.endswith(('.jpg','png')) ))
  
    # collect auto
    proctype = 'georeferenced_auto'
    data = collect(proctype)
    insert_into_db(db, 'auto', data)
    
    # collect perfect1
    proctype = 'georeferenced_perfect1_topos'
    data = collect(proctype)
    insert_into_db(db, 'perfect1', data)
    
    # collect perfect2
    proctype = 'georeferenced_perfect2_matches'
    data = collect(proctype)
    insert_into_db(db, 'perfect2', data)





##############

def getstats(proctype):
    db = sqlite3.connect('analyze/stats_baseline.db')
    stats = pg.VectorData()
    stats.fields = ['sim_id', 'rendopts', 'controlpoints', 'matched', 'segmentation', 'text', 'toponyms', 'transform', 'controlpoints_accuracy', 'accuracy', 'timings', 'log']
    for row in db.execute('select * from {}'.format(proctype)):
        row = list(row)
        stats.add_feature(row, None)

    # load json
    for fl in stats.fields[1:]: # skip sim_id (not json)
        stats.compute(fl, lambda f: json.loads(f[fl]))
    stats.compute('log', lambda f: f['log']['text'] if f['log'] else None)

    # all renderings
    stats_all = stats
    print stats_all

    # begin filtering down
    stats = stats_all

    # for some reason, log.txt files were also included, exclude these
    stats = stats.select(lambda f: not f['sim_id'].endswith('log.txt'))

    # exclude maps with nonlegible texts (jpg at 1000 pixel resolution)
    stats = stats.select(lambda f: not f['sim_id'].endswith('_6'))
    print stats
    
    return stats

stats = getstats('auto')
stats_p1 = getstats('perfect1')
stats_p2 = getstats('perfect2')

# scenes
scenes = set([f['sim_id'].split('_')[1] for f in stats])
print len(scenes)

# maps per scene
for scene in sorted(scenes):
    count = len(stats.select(lambda f: f['sim_id'].split('_')[1]==scene))
    print scene, count

fdafa







# visualize simulated scene footprints on global map
pg.vector.data.DEFAULT_SPATIAL_INDEX = 'quadtree'
m = pg.renderer.Map(3000,1500,crs='ESRI:54030')

# water
w = pg.VectorData()
top = [(x,90) for x in range(-180,180+1,1)]
bottom = [(x,-90) for x in reversed(range(-180,180+1,1))]
left = [(-180,y) for y in range(-90,90+1,1)]
right = [(180,y) for y in reversed(range(-90,90+1,1))]
geoj = {'type':'Polygon',
       'coordinates': [top+right+bottom+left]}
w.add_feature([], geoj)
m.add_layer(w, fillcolor=(0,118,190), fillopacity=0.3)

# countries
m.add_layer('data/ne_10m_admin_0_countries.shp',
           fillcolor=(72,191,145))

# footprints
d = pg.VectorData(crs=m.crs)
for scene in scenes:
    print(scene)
    r = pg.RasterData('maps/sim_{}_1_5_image.png'.format(scene))
    x1,y1,x2,y2 = r.bbox
    rect = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
    geoj = {'type':'Polygon',
           'coordinates':[rect]}
    _d = pg.VectorData(crs=r.crs)
    _d.add_feature([], geoj)
    _d = _d.manage.reproject(d.crs)
    geoj = _d[1].geometry
    d.add_feature([], geoj)
m.add_layer(d, outlinecolor='red', fillcolor='red', fillopacity=0.25)

m.save('analyze/figures/mapsim_footprints.png')







# misc exploration
import math
print stats
print stats.select(lambda f: f['accuracy'])
print stats.select(lambda f: f['accuracy'] and not math.isnan(f['accuracy']['max_georeferenced']['percent']))
isnan = stats.select(lambda f: f['accuracy'] and math.isnan(f['accuracy']['max_georeferenced']['percent']))

for f in stats.select(lambda f: f['accuracy']).aggregate(['sim_id'], fieldmapping=[('max',lambda f: f['accuracy']['max_georeferenced']['percent'],'max')]):
    print(f.row)
  
for f in stats.select(lambda f: f['sim_id']=='sim_75_12_5'):
    print f['accuracy']['max_georeferenced']['percent']
  
import math
import numpy as np
acc = [f['accuracy']['max_georeferenced']['percent'] * 100
       for f in stats.select(lambda f: f['accuracy'] 
                             #and f['accuracy']['max_georeferenced']['percent']
                             and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                            )]
for p in [0,50,75,90,95,99,100]:
    print p, '% -->', np.percentile(acc, p)
    
# histograms
def getacc(stats):
    acc = [f['accuracy']['max_georeferenced']['percent'] * 100
        for f in stats.select(lambda f: f['accuracy'] 
                             #and f['accuracy']['max_georeferenced']['percent']
                             and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                            )]
    acc = [a if a<100 else 100 for a in acc]
    return acc
def allhistos():
    fig = plt.figure()
    ax = plt.gca()
    import numpy as np
    
    y,bin_edges = np.histogram(getacc(stats), bins=100, density=True)
    x = [(x1+x2)/2.0 for x1,x2 in zip(bin_edges[:-1],bin_edges[1:])]
    ax.plot(x, y, label='auto')#, color='red')
    
    y,bin_edges = np.histogram(getacc(stats_p1), bins=100, density=True)
    x = [(x1+x2)/2.0 for x1,x2 in zip(bin_edges[:-1],bin_edges[1:])]
    ax.plot(x, y, label='perfect1')#, color='green')
    
    y,bin_edges = np.histogram(getacc(stats_p2), bins=100, density=True)
    x = [(x1+x2)/2.0 for x1,x2 in zip(bin_edges[:-1],bin_edges[1:])]
    ax.plot(x, y, label='perfect2')#, color='blue')
    
    #ax.set_xlim(0, 100000)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Max pixel error (% of image radius)')
    ax.set_ylabel('Share')
    #ax.set_ylabel('Maps')
    ax.legend()
    fig.savefig('analyze/figures/test.png')
allhistos()

# accu categoreis
def getacc(stats):
    acc = [f['accuracy']['max_georeferenced']['percent'] * 100
        for f in stats.select(lambda f: f['accuracy'] 
                             #and f['accuracy']['max_georeferenced']['percent']
                             and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                            )]
    return acc
def getaccgroups(data):
    import classypie as cp
    error_type = 'max'
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
    sub = data.copy()
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
                                    ]
                   )
    return agg
def allcats():
    fig = plt.figure()
    ax = plt.gca()
    import numpy as np
    
    #bins = [0,1,5,20,100,100000000000000]
    
    x,y = zip(*[(f['errclass'],f['count']) for f in getaccgroups(stats)])
    #y,bin_edges = np.histogram(getacc(stats), bins=bins)#, density=True)
    print y
    ytot = sum(y)
    y = [_y/float(ytot)*100 for _y in y]
    #y = [sum(y[:i+1]) for i in range(len(y))] # cumulative
    #x = [1,2,3,4,5]
    ax.plot(x, y, label='auto')#, color='red')
    
    x,y = zip(*[(f['errclass'],f['count']) for f in getaccgroups(stats_p1)])
    #y,bin_edges = np.histogram(getacc(stats_p1), bins=bins)#, density=True)
    print y
    ytot = sum(y)
    y = [_y/float(ytot)*100 for _y in y]
    #y = [sum(y[:i+1]) for i in range(len(y))] # cumulative
    #x = [1,2,3,4,5]
    ax.plot(x, y, label='perfect1')#, color='green')

    x,y = zip(*[(f['errclass'],f['count']) for f in getaccgroups(stats_p2)])
    #y,bin_edges = np.histogram(getacc(stats_p2), bins=bins)#, density=True)
    print y
    ytot = sum(y)
    y = [_y/float(ytot)*100 for _y in y]
    #y = [sum(y[:i+1]) for i in range(len(y))] # cumulative
    #x = [1,2,3,4,5]
    ax.plot(x, y, label='perfect2')#, color='blue')
    
    #ax.set_xlim(0, 100000)
    #ax.set_ylim(0, 1)
    ax.set_xlabel('Accuracy category (% of image radius)')
    ax.set_ylabel('% of maps')
    #ax.set_ylabel('Maps')
    x = [1,2,3,4,5,6]
    labels = ['0-1%','1-5%','5-20%','20-100%','>100%','Failed']
    plt.xticks(x, labels)
    ax.legend()
    fig.savefig('analyze/figures/test2.png')
allcats()
  
# ?
stats.select(lambda f: f['accuracy'] 
                       and not math.isnan(f['accuracy']['max_georeferenced']['percent'])
                      and f['accuracy']['max_georeferenced']['percent'] < 0.2
                      )






# NEW rmse loo vs true error
# farly poor correlation, gcp error can be both much lower and mucher higher than true error
import math
def err_gcp(f):
  if f['accuracy']:
    w = f['rendopts']['resolution']
    diag = math.hypot(w,w)
    perc = max(f['transform']['forward']['residuals']) / float(diag/2.0) * 100
    return perc
def err_true(f):
  if f['accuracy']:
    w = f['rendopts']['resolution']
    diag = math.hypot(w,w)
    perc = f['accuracy']['max_georeferenced']['pixels'] / float(diag/2.0) * 100
    return perc
def plot():
  x = [err_gcp(f) for f in stats]
  y = [err_true(f) for f in stats]
  #plt.xlim(0, 1500)#10)#20)#400)#1500)
  plt.xlabel('gcp % error')
  #plt.ylim(0, 100000)#1000)#20000)#100000)
  plt.ylabel('true % error')
  plt.scatter(x, y)
plot()





# NEW detected toponyms vs total toponyms
# scatterplot
alltopos = [f['rendopts']['placeopts']['quantity']
          for f in stats
         ]
topos = [len(f['toponyms']['features'])
          for f in stats
         ]
def plot():
  #plt.xlim()
  plt.xlabel('map toponyms')
  #plt.ylim(0,1000)#200000)
  plt.ylabel('toponyms detected')
  plt.scatter(alltopos, topos)
plot()

def plot(vals):
    plt.hist(vals, bins=50)
    plt.xlim(0, 1)
    
# detected
detected = [f['accuracy']['labels_detected']
            for f in stats
           if f['accuracy']]
plot(detected)

# matched
matched = [len(f['matched']['features']) / float( f['rendopts']['placeopts']['quantity'] )
            for f in stats
           if f['matched']]
plot(matched)

# used
used = [f['accuracy']['labels_used']
            for f in stats
           if f['accuracy']]
plot(used)

# stats
print 'detected', sum(detected)/float(len(detected))
print 'matched', sum(matched)/float(len(matched))
print 'used', sum(used)/float(len(used))





# NEW detected toponyms vs accuracy
# scatterplot
# fairly smooth reduction in error as toponyms increase, esp above 40
errors = [f['accuracy']['max_georeferenced']['percent'] * 100
          for f in stats
          if f['accuracy']
          #and f['accuracy']['max_georeferenced']['percent'] < 1
         ]
topos = [len(f['toponyms']['features'])
          for f in stats
          if f['accuracy']
         #and f['accuracy']['max_georeferenced']['percent'] < 1
         ]
def plot():
  #plt.xlim()
  plt.xlabel('toponyms')
  plt.ylim(0,1000)#200000)
  plt.ylabel('max error %')
  plt.scatter(topos, errors)
plot()
# boxplots (prob not necessary, just another way of showing above scatter and paper avg bar graph)
import classypie as cp
key = lambda f: f['accuracy']['max_georeferenced']['percent']*100
for g,items in cp.split('equal', key=key):
    #....
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
extents = [50, 10, 5, 1, 0.25] # ca 5000km, 1000km, 500km, 100km, and 10km
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
              #distributions=distributions,
              uncertainties=uncertainties,
              projections=projections,
              resolutions=resolutions,
              imformats=imformats,
              #metas=metas,
              #datas=datas,
                )
paramkeys = dict(extents=lambda f: f['rendopts']['regionopts']['extent'],
              toponyms=lambda f: f['rendopts']['placeopts']['quantity'],
              #distributions=lambda f: f['rendopts']['placeopts']['distribution'],
              uncertainties=lambda f: f['rendopts']['placeopts']['uncertainty'],
              projections=lambda f: f['rendopts']['projection'].split()[0].replace('+proj=',''),
              resolutions=lambda f: f['rendopts']['noiseopts']['resolution'],
              imformats=lambda f: f['rendopts']['noiseopts']['format'],
              #datas=lambda f: len(f['rendopts']['datas'])>1,
              #metas=lambda f: f['rendopts']['metaopts']['arealabels'],
                )
paramlabels = dict(extents='mapExtent (km)',
              toponyms='numToponyms',
              #distributions='toponymDispersed',
              uncertainties='toponymUncertainty (km)',
              projections='mapProjection',
              resolutions='imgResolution',
              imformats='pixelNoise',
              #metas='metaNoise',
              #datas='dataNoise',
                )

# map count per param
for k,func in paramkeys.items():
    statscopy = stats.select(lambda f: f['rendopts'])
    statscopy.compute('group', func)
    label = paramlabels[k]
    print(label)
    agg = statscopy.aggregate(['group'], fieldmapping=[('count','sim_id','count')])
    for f in agg:
        row = [''] + map(str, f.row)
        print ' & '.join(row) + ' \\\\'






        
# visualize heatmap with maxerr for each possible combination of param values

import numpy as np
params = paramvals.keys()
matsize = len([val for param in params for val in paramvals[param]])
mat = np.ones((matsize,matsize)) * float('nan')
errorpairs = []
goodthresh = 100
i1 = 0
for param1 in params:
    for val1 in paramvals[param1]:
        print(param1,val1)
        i2 = 0
        for param2 in params:
            for val2 in paramvals[param2]:
                if param1 != param2:
                    key1 = paramkeys[param1]
                    key2 = paramkeys[param2]
                    sub = stats.select(lambda f: key1(f)==val1 and key2(f)==val2)
                    errors = [f['accuracy']['percent']['max'] for f in sub 
                              if f['accuracy']]
                    errors = [err for err in errors
                             if not math.isnan(err)]
                    errors = [err for err in errors
                             if err<=goodthresh]
                    if errors:
                        percgood = (len(errors)/float(len(sub)))*100
                        mat[i1,i2] = percgood
                        #print('-->',param2,val2,percgood,len(errors))
                        errpair = param1,val1,param2,val2,percgood
                        errorpairs.append(errpair)
                i2 += 1
        i1 += 1

# list top worst combos

for errpair in sorted(errorpairs, key=lambda p: p[-1]):
    print(errpair)

# creat heatmap

def plot():
    import matplotlib.pyplot as plt
    plt.imshow(mat, interpolation='nearest')
    plt.grid(None)
    colorbar = plt.colorbar()
    colorbar.set_label('% usable maps')
    def valuelab(param,val):
        if param in 'extents uncertainties':
            return '{}km'.format(int(val*100))
        else:
            return str(val)
          
    # y labels
    labels = [str(valuelab(param,val))
              for param in params 
              for val in paramvals[param]]
    tlocs,tlabs = plt.yticks(range(len(labels)), labels)
    
    # y categories
    y = 0.2
    x = -7
    fontsize = tlabs[0].get_fontsize()
    for param in params:
        plt.annotate(param, (x,y), fontsize=fontsize, annotation_clip=False)
        y += len(paramvals[param])
    
    # x labels
    plt.gca().xaxis.tick_top()
    plt.xticks(range(len(labels)), labels, rotation=90)
    
    # x categories
    x = -0.5
    y = -4
    fontsize = tlabs[0].get_fontsize()
    for param in params:
        _param = {'uncertainties':'uncert.',
                'imformats':'img',
                'projections':'proj.',
                'resolutions':'res.'}.get(param, param)
        plt.annotate(_param, (x,y), fontsize=fontsize, annotation_clip=False)
        x += len(paramvals[param])
    
    plt.savefig('analyze/figures/errors_parampairs_heatmap.png',
               bbox_inches='tight', 
               pad_inches=0.1)
    
plot()

        
        




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
    print 'img size',img.size
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
    print truth.crs
    m = pg.renderer.Map(800,800,crs=truth.crs)
    m.add_layer(truth, transparency=0.5)
    georef = pg.RasterData(outfil)
    georef.mask = georef.bands[-1].compute('255-val').img # use alpha band as mask
    print georef.crs
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
#fig,axes = plt.subplots(3, 2, 
#                        figsize=(6,10))

metric = 'max'

# out of wack
examps = [f for f in stats if f['accuracy'] and 300 < f['accuracy']['percent'][metric] < 400]
for f in examps[5:6]:
    print f['accuracy']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_wack.png')
    #plot(f['sim_id'], 'wack')
    #subplot(axes[2,0], f['sim_id'], 'not usable\n'+'>100%')
# approx but unusable
examps = [f for f in stats if f['accuracy'] and 60 < f['accuracy']['percent'][metric] < 70]
for f in examps[4:5]:
    print f['accuracy']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_needsadj.png')
    #plot(f['sim_id'], 'approx_bad')
    #subplot(axes[1,1], f['sim_id'], 'approximate, needs fixing\n'+'>20%')
# approx
examps = [f for f in stats if f['accuracy'] and 10 < f['accuracy']['percent'][metric] < 12]
for f in examps[:1]:
    print f['accuracy']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_approx.png')
    #plot(f['sim_id'], 'approx')
    #subplot(axes[1,0], f['sim_id'], 'approximate\n'+'>5%')
# reasonable
examps = [f for f in stats if f['accuracy'] and 3 < f['accuracy']['percent'][metric] < 4]
for f in examps[:1]:
    print f['accuracy']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_reasonable.png')
    #plot(f['sim_id'], 'reasonable')
    #subplot(axes[0,1], f['sim_id'], 'reasonable\n'+'>1%')
# excellent
examps = [f for f in stats if f['accuracy'] and f['accuracy']['percent'][metric] < 0.7]
for f in examps[:1]:
    print f['accuracy']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_excellent.png')
    #plot(f['sim_id'], 'excellent')
    #subplot(axes[0,0], f['sim_id'], 'excellent\n'+'<1%')

#fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95,
#                wspace=0.05) #, hspace=0.05)
#fig.delaxes(axes[2,1])
#fig.savefig('analyze/figures/errors_maps_categories.png')

# table of overall simulation accuracy
def table(stats, error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        if f['accuracy'] and math.isnan(f['accuracy']['percent'][error_type]):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            v = f['accuracy']['percent'][error_type]
            return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
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

table(stats, 'max')
table(stats_p1, 'max')
table(stats_p2, 'max')

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






# FINAL: define error funcs
# either optional custom loo controlpoint error
'''
def geterror(f): 
  # model loo residuals max
  import automap as mapfit
  trans = mapfit.transforms.from_json(f['transform']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['controlpoints']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['controlpoints']['features']]
  resids = mapfit.accuracy.loo_residuals(trans, inpoints, outpoints, invert=True)
  err = max(resids)
  return err
def getdiag(f): 
  w = h = f['rendopts']['resolution']
  return math.hypot(w,h)
def getpercerror(f, error_type): 
  # ignore for now, always do max, see geterror()
  if f['accuracy']:
    err = geterror(f)
    perc = err / float(getdiag(f)/2.0) * 100
    return perc
'''

# or use the true surface error
'''fdfsdfs'''

# or universal geterror func
def get_gcp_resids(f): 
  # model loo residuals max
  import automap as mapfit
  trans = mapfit.transforms.from_json(f['transform']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['controlpoints']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['controlpoints']['features']]
  resids = mapfit.accuracy.residuals(trans, inpoints, outpoints, invert=True)
  return resids
def get_loo_resids(f): 
  # model loo residuals max
  import automap as mapfit
  trans = mapfit.transforms.from_json(f['transform']['backward'])
  inpoints = [(gcp['properties']['origx'],gcp['properties']['origy']) for gcp in f['controlpoints']['features']]
  outpoints = [(gcp['properties']['matchx'],gcp['properties']['matchy']) for gcp in f['controlpoints']['features']]
  resids = mapfit.accuracy.loo_residuals(trans, inpoints, outpoints, invert=True)
  return resids
def getdiag(f): 
  w = h = f['rendopts']['resolution']
  return math.hypot(w,h)
def getpercerror(f, error_type):
  if error_type == 'true_max':
    if f['accuracy']:
      return f['accuracy']['percent']['max']
  elif error_type == 'true_rmse':
    if f['accuracy']:
      return f['accuracy']['percent']['rmse']
  
  elif error_type == 'loo_max':
    if f['accuracy']:
      resids = get_loo_resids(f)
      err = max(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc
  elif error_type == 'loo_rmse':
    if f['accuracy']:
      resids = get_loo_resids(f)
      import automap as mapfit
      err = mapfit.accuracy.RMSE(resids)
      perc = err / float(getdiag(f)/2.0) * 100
      return perc
    
  elif error_type == 'gcp_max':
    if f['controlpoints_accuracy']:
      return f['controlpoints_accuracy']['percent']['max']
  elif error_type == 'gcp_rmse':
    if f['controlpoints_accuracy']:
      return f['controlpoints_accuracy']['percent']['rmse']


    
    
    
    

# table of accuracy for realistic quality maps
def table(stats, error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        v = getpercerror(f,error_type)
        if f['accuracy'] and math.isnan(v):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.copy()
    sub = sub.select(lambda f: f['rendopts'] and \
                   #f['rendopts']['noiseopts']['resolution']>1500 and \
                   #f['rendopts']['noiseopts']['format']=='png' and \
                   f['rendopts']['placeopts']['quantity']>=20 and \
                   f['rendopts']['placeopts']['uncertainty']<=0.1 and \
                   #f['rendopts']['placeopts']['distribution']=='random' and \
                   #f['rendopts']['projection']==None and \
                   #f['rendopts']['metaopts']['arealabels']==False and \
                   f['rendopts']['regionopts']['extent']>=1 and \
                     True \
                  )
    sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.5 and \
                     f['rendopts']['regionopts']['extent']<=1 \
                                                     )
                    )
    sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']>=0.1 and \
                     f['rendopts']['regionopts']['extent']==0.25 \
                                                     )
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

table(stats, 'max')
table(stats_p1, 'max')
table(stats_p2, 'max')








# FINAL: table of accuracy for multiple types of maps

import numpy as np
def table(stats, error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        v = getpercerror(f,error_type)
        if f['accuracy'] and math.isnan(v):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.copy()
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
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
  rows = table(stats, 'true_max')
  rows = [[row[0]]+row[2:] for row in rows]
  addrows = table(stats, 'loo_max')
  for addrow in addrows:
    for row in rows:
      if addrow[0] == row[0]:
        row.extend(addrow[2:])
        break
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
      print latexrow

# exclude outliers
sub = stats.copy()
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.5 and \
                     f['rendopts']['regionopts']['extent']==1 \
                                                     )
                    )
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.5 and \
                     f['rendopts']['regionopts']['extent']==0.25 \
                                                     )
                    )
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.1 and \
                     f['rendopts']['regionopts']['extent']==0.25 \
                                                     )
                    )

## 1

# make table for all
print sub
tables(sub)
#table(sub, 'true_max')
#table(sub, 'loo_max')
#table(sub, 'gcp_max')

## 2

# select realistic (should actually appear last)
sub = sub.select(lambda f: f['rendopts'] and \
               #f['rendopts']['noiseopts']['resolution']>1500 and \
               #f['rendopts']['noiseopts']['format']=='png' and \
               f['rendopts']['placeopts']['quantity']>=20 and \
               f['rendopts']['placeopts']['uncertainty']<=0.1 and \
               #f['rendopts']['placeopts']['distribution']=='random' and \
               #f['rendopts']['projection']==None and \
               #f['rendopts']['metaopts']['arealabels']==False and \
               #f['rendopts']['regionopts']['extent']>=1 and \
                 True \
              )

# make table for realistic
print sub
tables(sub)
#table(sub, 'true_max')
#table(sub, 'loo_max')
#table(sub, 'gcp_max')

## 3

# select good-quality
sub = sub.select(lambda f: f['rendopts'] and \
               f['rendopts']['noiseopts']['resolution']>1500 and \
               #f['rendopts']['noiseopts']['format']=='png' and \
               f['rendopts']['placeopts']['quantity']>=20 and \
               f['rendopts']['placeopts']['uncertainty']<=0.1 and \
               #f['rendopts']['placeopts']['distribution']=='random' and \
               #f['rendopts']['projection']==None and \
               #f['rendopts']['metaopts']['arealabels']==False and \
               #f['rendopts']['regionopts']['extent']>=1 and \
                 True \
              )

# make table for good-quality
print sub
tables(sub)
#table(sub, 'true_max')
#table(sub, 'loo_max')
#table(sub, 'gcp_max')








# FINAL: get polynomial stats

## 1 all

sub = stats.select(lambda f: f['transform'])
key = lambda f: f['transform']['forward']['params']['order']
agg = sub.aggregate(key=key,
                        fieldmapping=[('polynomial',key,'first'),
                                      ('count','sim_id','count'),
                                      ('avg_max_err',lambda f: getpercerror(f, 'true_max'),'mean'),
                                    ]
                   )
for f in agg:
    order,count,avgerr = f.row
    perc = count / float(len(sub)) * 100
    print(order, count, perc, avgerr)
    
## 2

# select realistic (should actually appear last)
sub = stats.select(lambda f: f['transform'])
sub = sub.select(lambda f: f['rendopts'] and \
               #f['rendopts']['noiseopts']['resolution']>1500 and \
               #f['rendopts']['noiseopts']['format']=='png' and \
               f['rendopts']['placeopts']['quantity']>=20 and \
               f['rendopts']['placeopts']['uncertainty']<=0.1 and \
               #f['rendopts']['placeopts']['distribution']=='random' and \
               #f['rendopts']['projection']==None and \
               #f['rendopts']['metaopts']['arealabels']==False and \
               #f['rendopts']['regionopts']['extent']>=1 and \
                 True \
              )
key = lambda f: f['transform']['forward']['params']['order']
agg = sub.aggregate(key=key,
                        fieldmapping=[('polynomial',key,'first'),
                                      ('count','sim_id','count'),
                                      ('avg_max_err',lambda f: getpercerror(f, 'true_max'),'mean'),
                                    ]
                   )
for f in agg:
    order,count,avgerr = f.row
    perc = count / float(len(sub)) * 100
    print(order, count, perc, avgerr)
    
## 3

# select good-quality
sub = stats.select(lambda f: f['transform'])
sub = sub.select(lambda f: f['rendopts'] and \
               f['rendopts']['noiseopts']['resolution']>1500 and \
               #f['rendopts']['noiseopts']['format']=='png' and \
               f['rendopts']['placeopts']['quantity']>=20 and \
               f['rendopts']['placeopts']['uncertainty']<=0.1 and \
               #f['rendopts']['placeopts']['distribution']=='random' and \
               #f['rendopts']['projection']==None and \
               #f['rendopts']['metaopts']['arealabels']==False and \
               #f['rendopts']['regionopts']['extent']>=1 and \
                 True \
              )
key = lambda f: f['transform']['forward']['params']['order']
agg = sub.aggregate(key=key,
                        fieldmapping=[('polynomial',key,'first'),
                                      ('count','sim_id','count'),
                                      ('avg_max_err',lambda f: getpercerror(f, 'true_max'),'mean'),
                                    ]
                   )
for f in agg:
    order,count,avgerr = f.row
    perc = count / float(len(sub)) * 100
    print(order, count, perc, avgerr)
    
## AND FINALLY DIVIDED INTO EACH ACCURACY CATEGORY (full sample)

sub = stats.select(lambda f: f['transform'])
import classypie as cp
def classfunc(f): 
    v = getpercerror(f,'true_max')
    if f['accuracy'] and math.isnan(v):
        # nan accuracy (geod dist out-of-this-world)
        # group with not-usable category
        return 5
    elif f['accuracy']:
        return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
    else:
        return 6
sub.compute('errclass', classfunc)

key = lambda f: f['transform']['forward']['params']['order']
for order,group in sub.manage.split(key):
    print(order)
    
    agg = group.aggregate(key=[]'errclass'],
                            fieldmapping=[('count','sim_id','count'),
                                          ('avg_max_err',lambda f: getpercerror(f, 'true_max'),'mean'),
                                        ]
                       )
    for f in agg:
        errclass,count,avgerr = f.row
        perc = count / float(len(group)) * 100
        print(errclass, count, perc, avgerr)
    







# FINAL: acc. linegraphs comparing algo stages/assumptions
import numpy as np
def table(stats, error_type='max'):
    # x different centers
    import classypie as cp
    def classfunc(f): 
        v = getpercerror(f,error_type)
        if f['accuracy'] and math.isnan(v):
            # nan accuracy (geod dist out-of-this-world)
            # group with not-usable category
            return 5
        elif f['accuracy']:
            return cp.find_class(v, [0, 1, 5, 20, 100, 110000000])[0] if not math.isnan(v) else None
        else:
            return 6
    sub = stats.copy()
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
                                    ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    cumul = 0
    rows = []
    for f in agg:
        cumul += f.row[-1]
        row = f.row + [cumul]
        rows.append(row)
    return rows 

  
def subset(stats):
    # exclude outliers
    sub = stats.copy()
    sub = sub.select(lambda f: f['rendopts'] and not (\
                         f['rendopts']['placeopts']['uncertainty']==0.5 and \
                         f['rendopts']['regionopts']['extent']==1 \
                                                         )
                        )
    sub = sub.select(lambda f: f['rendopts'] and not (\
                         f['rendopts']['placeopts']['uncertainty']==0.5 and \
                         f['rendopts']['regionopts']['extent']==0.25 \
                                                         )
                        )
    sub = sub.select(lambda f: f['rendopts'] and not (\
                         f['rendopts']['placeopts']['uncertainty']==0.1 and \
                         f['rendopts']['regionopts']['extent']==0.25 \
                                                         )
                        )

    # select realistic
    sub = sub.select(lambda f: f['rendopts'] and \
                   #f['rendopts']['noiseopts']['resolution']>1500 and \
                   #f['rendopts']['noiseopts']['format']=='png' and \
                   f['rendopts']['placeopts']['quantity']>=20 and \
                   f['rendopts']['placeopts']['uncertainty']<=0.1 and \
                   #f['rendopts']['placeopts']['distribution']=='random' and \
                   #f['rendopts']['projection']==None and \
                   #f['rendopts']['metaopts']['arealabels']==False and \
                   #f['rendopts']['regionopts']['extent']>=1 and \
                     True \
                  )

    return sub

  
def plot():
    # full
    sub = subset(stats)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    x = [i for i in range(len(y))]
    plt.plot(x, y,
           label='Cur. methodology',
            linewidth=2,
            marker='o',
            color='tab:orange',
           )
    
    # perfect toponyms
    sub = subset(stats_p1)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    x = [i for i in range(len(y))]
    plt.plot(x, y,
           label='Perf. toponyms',
            linewidth=2,
            marker='o',
            color='tab:blue',
           )
    
    # perfect matching
    sub = subset(stats_p2)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    x = [i for i in range(len(y))]
    plt.plot(x, y,
           label='Perf. matching',
            linewidth=2,
            marker='o',
            color='tab:green',
           )
    
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xticks(x, ['Exc.','Reas.','Approx.','Needs.','Not.','Failed'])
    plt.xlabel('Accuracy category')
    plt.ylabel('% of maps')
    
    plt.savefig('analyze/figures/outcomes_compare_perfect.png')
  
plot()

def plot():
    # full
    sub = subset(stats)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    ycum = np.cumsum(y)
    x = [i for i in range(len(y))]
    print x
    plt.plot(x, ycum,
           label='Cur. methodology',
            linewidth=2,
            marker='o',
            color='tab:orange',
           )
    
    # perfect toponyms
    sub = subset(stats_p1)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    ycum = np.cumsum(y)
    x = [i for i in range(len(y))]
    plt.plot(x, ycum,
           label='Perf. toponyms',
            linewidth=2,
            marker='o',
            color='tab:blue',
           )
    
    # perfect matching
    sub = subset(stats_p2)
    print sub
    tab = table(sub, 'true_max')
    y = [row[2] for row in tab]
    ycum = np.cumsum(y)
    x = [i for i in range(len(y))]
    plt.plot(x, ycum,
           label='Perf. matching',
            linewidth=2,
            marker='o',
            color='tab:green',
           )
    
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xticks(x, ['Exc.','Reas.','Approx.','Needs.','Not.','Failed'])
    plt.xlabel('Cumul. accuracy category')
    plt.ylabel('% of maps')
    plt.ylim(0,105)
    
    plt.savefig('analyze/figures/outcomes_compare_perfect_cumul.png')
  
plot()









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

    
  
# same but linegraphs of success rate (<100)

import numpy as np

# linegraphs for each param value
def outcomeplot_stats(ax, stats, param, vals, xlabel=True, **baselinevals):
    print param
    #ax.set_title(param)
    
    if xlabel:
        paramlabel = paramlabels[param]
        ax.set_xlabel(paramlabel, weight='bold')
    if axes.ndim == 1 and ax in (axes[0],axes[3]):
        ax.set_ylabel('% of maps')
        
    rates1 = []
    rates2 = []
    def is_param_baseline(f):
        # NO LONGER SETS ALL TO BASELINE, ONLY CUSTOM BASELINE VALS
        return all([paramkeys[k](f)==v for k,v in baselinevals.items()])
    for val in vals:
        sub = stats.select(lambda f: f['rendopts'] and is_param_baseline(f) and paramkeys[param](f) == val)
        outcomes = np.array([f['accuracy']['percent']['max'] if f['accuracy'] and not math.isnan(f['accuracy']['percent']['max']) else 999999999999999
                            for f in sub])
        #rate1 = (outcomes!=999999999999999).mean() * 100 # not fail
        rate1 = (outcomes<100).mean() * 100 # usable
        rate2 = (outcomes<5).mean() * 100 # human-equivalent
        print 'stats for {}={} (n={})'.format(param, val, len(outcomes))
        print rate1,rate2 #,rate3
        rates1.append(rate1)
        rates2.append(rate2)
        
    # get labels
    labelvals = list(vals)
    if isinstance(vals[0], basestring):
        # only set tick labels for string values
        vals = [i for i,v in enumerate(vals)]
        ax.set_xticks(vals)
        ax.set_xticklabels(labelvals)
    else:
        if param in 'extents uncertainties':
            vals = [int(v*100) for v in vals]
    
    # usable
    ax.plot(vals, rates2,
           label='High-accuracy (<5% error)',
            linewidth=2,
            marker='o',
            color='tab:green',
           )
    
    # human
    ax.plot(vals, rates1,
           label='Low-accuracy (<100% error)',
            linestyle='dashed',
            linewidth=2,
            marker='o',
            color='tab:blue',
           )

# exclude outliers
sub = stats.copy()
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.5 and \
                     f['rendopts']['regionopts']['extent']==1 \
                                                     )
                    )
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.5 and \
                     f['rendopts']['regionopts']['extent']==0.25 \
                                                     )
                    )
sub = sub.select(lambda f: f['rendopts'] and not (\
                     f['rendopts']['placeopts']['uncertainty']==0.1 and \
                     f['rendopts']['regionopts']['extent']==0.25 \
                                                     )
                    )
    
fig,axes = plt.subplots(2, 3, 
                        sharey=True, #sharex=True, sharey=True,
                        figsize=(8,6))
axes = axes.flatten()
    
outcomeplot_stats(axes[0], sub, 'extents', extents, )

outcomeplot_stats(axes[1], sub, 'resolutions', resolutions, )

outcomeplot_stats(axes[2], sub, 'projections', projections, )

outcomeplot_stats(axes[3], sub, 'imformats', imformats, )

outcomeplot_stats(axes[4], sub, 'toponyms', quantities, )

outcomeplot_stats(axes[5], sub, 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15,
                   hspace=0.4) #wspace=0.05)

handles, labels = axes[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
handles = [Line2D([0],[0],linewidth=2,
                  marker=None,
                  color='tab:green',),
           Line2D([0],[0],linewidth=2,
                  marker=None,
                  linestyle='dashed',
                  color='tab:blue',),
          ]
fig.legend(handles, labels, ncol=2, frameon=False,
          loc='lower center')

fig.savefig('analyze/figures/outcomes_baseline_lineplots.png')
  
  
  
# same but bargraphs of success rate (<100)

import numpy as np

# bargraphs for each param value
def outcomeplot_stats(ax, param, vals, xlabel=True, **baselinevals):
    print param
    #ax.set_title(param)
    if xlabel:
        paramlabel = paramlabels[param]
        ax.set_xlabel(paramlabel, weight='bold')
    if axes.ndim == 1 and ax in (axes[0],axes[3]):
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
        outcomes = np.array([f['accuracy']['percent']['max'] if f['accuracy'] and not math.isnan(f['accuracy']['percent']['max']) else 999999999999999
                            for f in sub])
        #rate1 = (outcomes!=999999999999999).mean() * 100 # not fail
        rate1 = (outcomes<100).mean() * 100 # usable
        #rate2 = (outcomes<20).mean() * 100 # no fixing
        rate2 = (outcomes<5).mean() * 100 # human-equivalent
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
               ##color='tab:blue', 
               width=w, 
               label='Usable (<100% error)',
                )
    _ = ax.bar([x+w for x in range(len(rates2))], rates2, 
               tick_label=[str(v)[:12] for v in labelvals], 
               align='edge',
               ##color='tab:orange', 
               width=w, 
               label='Human equivalent (<5% error)',
                )
    #_ = ax.bar([x+w+w for x in range(len(rates3))], rates3, 
    #           #tick_label=[str(v)[:12] for v in labelvals], align='edge',
    #           color='tab:green', width=w, 
    #            )

fig,axes = plt.subplots(2, 3, 
                        sharey=True, #sharex=True, sharey=True,
                        figsize=(8,6))
axes = axes.flatten()
    
outcomeplot_stats(axes[0], 'extents', extents, )

outcomeplot_stats(axes[1], 'resolutions', resolutions, )

outcomeplot_stats(axes[2], 'projections', projections, )

##outcomeplot_stats(axes[3], 'datas', datas, )

##outcomeplot_stats(axes[4], 'metas', metas, )

outcomeplot_stats(axes[3], 'imformats', imformats, )

outcomeplot_stats(axes[4], 'toponyms', quantities, )

outcomeplot_stats(axes[5], 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15,
                   hspace=0.4) #wspace=0.05)
#fig.delaxes(axes[-1])
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
    
    
    
