import pythongis as pg
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math
import sqlite3

import automap as mapfit

# NOTES:
# - take all permutations
# - associate each with its avg error metric
# - use error metric to cluster combinations of parameter values/ranges that result in lots of error
# - thus can know which types of maps are more suitable, and which method params work best




print(os.getcwd())
try:
    os.chdir('simulations')
except:
    pass




# Loop and create table
if False:
    out = pg.VectorData()
    out.fields = ['sim_id', 'rendopts', 'controlpoints', 'matched', 'segmentation', 'text', 'toponyms', 'transform', 'accuracy', 'timings', 'log']

    roots = set(('_'.join(fil.split('_')[:4]) for fil in os.listdir('maps')))

    for root in roots:
        print root

        vals = {'sim_id':root}

        path = 'maps/{}_opts.json'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['rendopts'] = dat
                
        path = 'output/{}_georeferenced_auto_controlpoints.geojson'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['controlpoints'] = dat
                
        path = 'output/{}_georeferenced_auto_debug_gcps_matched.geojson'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['matched'] = dat
                
        path = 'output/{}_georeferenced_auto_debug_segmentation.geojson'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['segmentation'] = dat
                
        path = 'output/{}_georeferenced_auto_debug_text.geojson'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['text'] = dat
                
        path = 'output/{}_georeferenced_auto_debug_text_toponyms.geojson'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['toponyms'] = dat
                
        path = 'output/{}_georeferenced_auto_transform.json'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['transform'] = dat

        path = 'output/{}_georeferenced_auto_error.json'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['accuracy'] = dat
                
        path = 'output/{}_georeferenced_auto_debug_timings.json'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = json.load(fobj)
                vals['timings'] = dat
                
        path = 'output/{}_georeferenced_auto_log.txt'.format(root)
        if os.path.lexists(path):
            with open(path) as fobj:
                dat = fobj.read()
                vals['log'] = {'text':dat}

        #print vals
        
        out.add_feature(vals, None)


    # encode dicts to json for saving
    print 'json dumping'
    for fl in out.fields[1:]: # skip sim_id (not json)
        out.compute(fl, lambda f: json.dumps(f[fl]))

    # save
    #out.save('analyze/stats.csv')
    db = sqlite3.connect('analyze/stats.db')
    cur = db.cursor()
    fielddef = ','.join(['{} text'.format(fl) for fl in out.fields])
    cur.execute('create table data ({})'.format(fielddef))
    for f in out:
      qs = ','.join('?'*len(out.fields))
      vals = tuple(f.row)
      cur.execute('insert into data values ({})'.format(qs), vals)
    db.commit()





##############

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

# exclude maps with nonlegible texts (jpg at 1000 pixel resolution)
stats = stats.select(lambda f: not f['sim_id'].endswith('_6'))
    
  
  
  
fdsafsa





                





    
    
    
### EXPLORATION

sns.set_style("darkgrid")

# basics

# test number of georeferences actually attempted
'attempted:', len(stats.select(lambda f: f['log']))
# test number of georeferences that didn't complete
'failed:', len(stats.select(lambda f: not f['timings']))
# test number of georeferences that raised error
'errored:', len(stats.select(lambda f: f['log'] and 'Error:' in f['log']))
# list errors
def geterror(f): 
    text = f['log']
    if text:
        return text[text.find('Error:'):text.find('\n', text.find('Error:'))]
erragg = stats.aggregate(geterror, 
                         fieldmapping=[('error',geterror,'first'),
                                       ('count','sim_id','count')])
erragg.sort(lambda f: -f['count'])
for f in erragg:
    print f['count'], f['error']
# test number of georeferences that timed out
'timeouts', len(stats.select(lambda f: f['log'] and 'total runtime: ' not in f['log'] and 'Error:' not in f['log']))


# timings
timings = [f['timings']['total'] for f in stats if f['timings']]
mean = sum(timings)/float(len(timings))
med = sorted(timings)[len(timings)//2]
'total time avg', mean, 'median', med, 'max', max(timings)
plt.hist(timings, bins=1000)
plt.show()
'below 5 mins', len([t for t in timings if t < 300]) / float(len(timings))


# overall accuracy distribution across all tests
errors = [f['accuracy']['max_georeferenced']['percent'] for f in stats if f['accuracy']]
nans = len([e for e in errors if math.isnan(e)])
'error nan count', nans
errors = [e for e in errors if not math.isnan(e)]
mean = sum(errors)/float(len(errors))
med = sorted(errors)[len(errors)//2]
'true max error (fraction of img radius): avg', mean, 'median', med, 'max', max(errors)
# cumulative
plt.hist(errors, bins=100, cumulative=True, normed=True)
plt.show()
# all
plt.hist(errors, bins=100, range=(0.01,1))
plt.show()

# catgories
'out of wack, >100%', len([e for e in errors if e > 1]) / float(len(errors))
'approx but unusable, >20%', len([e for e in errors if 0.2 < e < 1]) / float(len(errors))
'approx, >5%', len([e for e in errors if 0.05 < e < 0.2]) / float(len(errors))
'reasonable, >1%', len([e for e in errors if 0.01 < e < 0.05]) / float(len(errors))
'excellent, <1%', len([e for e in errors if e < 0.01]) / float(len(errors))


# visualize examples? 
from IPython.display import display
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
def compare_maps(sim_id):
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
# out of wack
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] > 1]
for f in examps[:2]:
    print f['accuracy']['max_georeferenced']
    img = compare_maps(f['sim_id'])
    display(img)
# approx but unusable
examps = [f for f in stats if f['accuracy'] and 0.2 < f['accuracy']['max_georeferenced']['percent'] < 1]
for f in examps[:2]:
    print f['accuracy']['max_georeferenced']
    img = compare_maps(f['sim_id'])
    display(img)
# approx
examps = [f for f in stats if f['accuracy'] and 0.05 < f['accuracy']['max_georeferenced']['percent'] < 0.2]
for f in examps[:2]:
    print f['accuracy']['max_georeferenced']
    img = compare_maps(f['sim_id'])
    display(img)
# reasonable
examps = [f for f in stats if f['accuracy'] and 0.01 < f['accuracy']['max_georeferenced']['percent'] < 0.05]
for f in examps[:2]:
    print f['accuracy']['max_georeferenced']
    img = compare_maps(f['sim_id'])
    display(img)
# excellent
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] < 0.01]
for f in examps[:2]:
    print f['accuracy']['max_georeferenced']
    img = compare_maps(f['sim_id'])
    display(img)


# define params
extents = [50, 10, 1, 0.1] # ca 5000km, 1000km, 100km, and 10km
quantities = [80, 40, 20, 10]
distributions = ['dispersed', 'random'] # IMPROVE W NUMERIC
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
projections = [None, # lat/lon
               '+proj=robin +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54030', # Robinson
               ]
resolutions = [3000, 2000, 1000] 
imformats = ['png','jpg']
metas = [{'title':'','legend':False,'arealabels':False,'legendoptions':{},'titleoptions':{}}, # nothing
         {'title':'This is the Map Title','titleoptions':{'fillcolor':'white'},'legend':True,'legendoptions':{'fillcolor':'white'},'arealabels':True}, # text noise + meta boxes (arealabels + title + legend)
         ]
allparams = dict(extents=extents,
              quantities=quantities,
              distributions=distributions,
              uncertainties=uncertainties,
              projections=projections,
              resolutions=resolutions,
              imformats=imformats,
              metas=metas)
  
  
# boxplots for each param value
def printstats(errors):
    import numpy as np
    errors = np.array(errors)
    print '-> median', np.median(errors)
    print '-> mean', errors.mean()
    print '-> max', errors.max()

def boxplot_stats(param, vals, getvalfunc):
    fig = plt.figure()
    plt.title(param)
    plt.ylabel('max true error (as % of img radius)')
    valerrors = []
    for val in vals:
        sub = stats.select(lambda f: f['rendopts'] and f['accuracy'] and getvalfunc(f) == val)
        errors = [f['accuracy']['max_georeferenced']['percent'] for f in sub]
        errors = [e for e in errors if not math.isnan(e)]
        print 'stats for {}={} (n={})'.format(param, val, len(errors))
        printstats(errors)
        valerrors.append(errors)
    _ = plt.boxplot(valerrors, labels=[str(v)[:30] for v in vals],
                   showcaps=False, showfliers=False)
    plt.show()
    
def scatter(param, getvalfunc):
    fig = plt.figure()
    plt.title(param)
    plt.ylabel('max true error (as % of img radius)')
    sub = stats.select(lambda f: f['accuracy'] and getvalfunc(f) is not None)
    x = [getvalfunc(f) for f in sub]
    y = [f['accuracy']['max_georeferenced']['percent'] for f in sub]
    _ = plt.scatter(x, y)
    plt.show()

boxplot_stats('extents', extents, lambda f: f['rendopts']['regionopts']['extent'])

boxplot_stats('resolutions', resolutions, lambda f: f['rendopts']['noiseopts']['resolution'])

boxplot_stats('projections', projections, lambda f: f['rendopts']['projection'])

boxplot_stats('imformats', imformats, lambda f: f['rendopts']['noiseopts']['format'])

boxplot_stats('metas', metas, lambda f: f['rendopts']['metaopts'])

boxplot_stats('quantities', quantities, lambda f: f['rendopts']['placeopts']['quantity'])

boxplot_stats('distributions', distributions, lambda f: f['rendopts']['placeopts']['distribution'])

boxplot_stats('uncertainties', uncertainties, lambda f: f['rendopts']['placeopts']['uncertainty'])

# and special stuff

scatter('labels', lambda f: f['accuracy'] and f['accuracy']['labels'])

scatter('labels detected', lambda f: f['accuracy'] and f['accuracy']['labels'] * f['accuracy']['labels_detected'])

scatter('labels used', lambda f: f['accuracy'] and f['accuracy']['labels'] * f['accuracy']['labels_used'])

scatter('labels scat perc', lambda f: f['accuracy'] and f['accuracy']['labels_used_scat_perc'])


# look at ideal map quality
sub = stats.select(lambda f: f['rendopts'] and f['accuracy'])
sub = sub.select(lambda f: f['rendopts']['noiseopts']['resolution']>500 and \
                   #f['rendopts']['noiseopts']['format']=='png' and \
                   f['rendopts']['placeopts']['quantity']>10 and \
                   f['rendopts']['placeopts']['uncertainty']<=0.10 and \
                   #f['rendopts']['placeopts']['distribution']=='dispersed' and \
                   f['rendopts']['regionopts']['extent']>=1 and \
                   f['rendopts']['projection']==None and \
                   f['rendopts']['metaopts']['arealabels']==False
                  )
def table():
    # x different centers
    import classypie as cp
    classfunc = lambda v: cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 11000])[0] if not math.isnan(v) else None
    agg = sub.aggregate(key=lambda f: classfunc(f['accuracy']['max_georeferenced']['percent']),
                    fieldmapping=[('class',lambda f: classfunc(f['accuracy']['max_georeferenced']['percent']),'first'),
                                  ('count',lambda f: f['sim_id'],'count'),
                                 ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    for f in agg:
        print f.row
        
table()


# NOT FINISHED: determine ideal config and keep constant
# then, for each param, render boxplot for each value
sub = stats.select(lambda f: f['rendopts'])
sub = sub.select(lambda f: f['rendopts']['noiseopts']['resolution']>500 and \
                   #f['rendopts']['noiseopts']['format']=='png' and \
                   #f['rendopts']['placeopts']['quantity']==80 and \
                   f['rendopts']['placeopts']['uncertainty']==0 and \
                   #f['rendopts']['placeopts']['distribution']=='dispersed' and \
                   #f['rendopts']['regionopts']['extent']==10 and \
                   f['rendopts']['projection']==None and \
                   f['rendopts']['metaopts']['arealabels']==False
                  )
errors = [f['accuracy']['rmse_georeferenced']['percent'] for f in sub if f['accuracy']]
errors = [e for e in errors if not math.isnan(e)]
'below 100%', len([e for e in errors if e < 1]) / float(len(sub))
# cumulative
plt.hist(errors, bins=100, cumulative=True, normed=True)
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
### JOURNAL ARTICLE

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
    ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 10deg ie 1000km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==10)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[1]
    ax.set_title('1000 km extent')
    ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 1deg ie 100km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==1)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[2]
    ax.set_title('100 km extent')
    ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # 0.1deg ie 10km
    sub = stats.select(lambda f: f['rendopts'] and f['rendopts']['regionopts']['extent']==0.1)
    errors = [f['accuracy'][error_type+'_georeferenced']['geographic'] for f in sub if f['accuracy']]
    errors = [e for e in errors if not math.isnan(e)]
    ax = axes[3]
    ax.set_title('10 km extent')
    ax.set_yscale('log')
    ax.set_xlabel(error_type+' km error')
    #ax.set_ylabel('# maps')
    ax.hist(errors, bins=bins, range=range) #, range=(0,5000))
    
    # save
    plt.savefig('analyze/figures/error_km_by_extent.png')
    
plot()

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

# example map error, error surface vs map overlay
from IPython.display import display
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

# out of wack
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] > 1]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    plot(f['sim_id'], 'wack')
# approx but unusable
examps = [f for f in stats if f['accuracy'] and 0.5 < f['accuracy']['max_georeferenced']['percent'] < 0.6]
for f in examps[2:3]:
    print f['accuracy']['max_georeferenced']
    plot(f['sim_id'], 'approx_bad')
# approx
examps = [f for f in stats if f['accuracy'] and 0.1 < f['accuracy']['max_georeferenced']['percent'] < 0.12]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    plot(f['sim_id'], 'approx')
# reasonable
examps = [f for f in stats if f['accuracy'] and 0.01 < f['accuracy']['max_georeferenced']['percent'] < 0.05]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    plot(f['sim_id'], 'reasonable')
# excellent
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] < 0.01]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    plot(f['sim_id'], 'excellent')

# table of overall simulation accuracy
def table():
    # x different centers
    import classypie as cp
    def classfunc(f): 
        v = f['accuracy']['max_georeferenced']['percent']
        return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0] if not math.isnan(v) else None
    sub = stats.select(lambda f: f['accuracy'])
    sub = sub.select(lambda f: not math.isnan(f['accuracy']['max_georeferenced']['percent']))
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                        fieldmapping=[('count','sim_id','count'),
                                    ]
                   )
    agg.compute('perc', lambda f: f['count']/float(len(sub))*100)
    cumul = 0
    for f in agg:
        cumul += f.row[-1]
        print f.row + [cumul]
        
    print ['total success', len(sub), len(sub)/float(len(stats))*100, 'NA']
    print ['total failure', len(stats)-len(sub), (len(stats)-len(sub))/float(len(stats))*100, 'NA']
    print ['total', len(sub), 100, 'NA']

table()

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
                            fieldmapping=[('total',k,'sum'),
                                          ('avg',k,'avg'),
                                          ('median',k,lambda vals: sorted(vals)[len(vals)//2]),
                                          ]
                            )
        f = agg[1]
        row = [k] + [v for v in f.row]
        rows.append(row)
    
    total = sum((r[1] for r in rows[:-1]))
    for row in rows:
        perc = row[1] / float(total) * 100
        row.insert(2, perc)
        row = [round(v, 1) if isinstance(v,float) else v
               for v in row]
        #print row
        latexrow = ' & '.join(map(str,row)) + ' \\\\'
        print latexrow

table()

# table of accuracy for good quality maps
# ... 

# boxplots of good quality maps, adjusting one parameter at a time
# ... 
    
# table of error sources
def table():
    import classypie as cp
    def classfunc(f): 
        if f['accuracy']:
            v = f['accuracy']['max_georeferenced']['percent']
            if v==None or math.isnan(v):
                return 6
            else:
                return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0]
    sub = stats.select(lambda f: f['rendopts'])
    print sub
    sub.compute('errclass', classfunc)
    agg = sub.aggregate(key=['errclass'],
                    fieldmapping=[#('count',lambda f: 1,'count'),
                                  ('extent',lambda f: f['rendopts']['regionopts']['extent']*100,'avg'),
                                  ('resolution',lambda f: f['rendopts']['resolution'],'avg'),
                                  ('quantity',lambda f: f['rendopts']['placeopts']['quantity'],'avg'),
                                  ('uncertainty',lambda f: f['rendopts']['placeopts']['uncertainty']*100,'avg'),
                                  ('gcps',lambda f: f['accuracy']['labels_used'] * float(f['rendopts']['placeopts']['quantity']) if f['accuracy'] and f['rendopts']['placeopts']['quantity'] else None,'avg'),
                                  #distribution
                                  ('dispersed', lambda f: 1 if f['rendopts']['placeopts']['distribution'] == 'dispersed' else 0, 'avg'),
                                  #projection
                                  ('projection', lambda f: 1 if f['rendopts']['projection'] else 0, 'avg'),
                                  #data noise (not available...)
                                  #text noise
                                  ('metanoise', lambda f: 1 if f['rendopts']['metaopts']['legend'] else 0, 'avg'),
                                  #pixel noise
                                  ('formatnoise', lambda f: 1 if f['rendopts']['noiseopts']['format']=='jpg' else 0, 'avg'),
                                 ]
                   )
    for f in agg:
        yield f
         
for f in table():
    latexrow = '{errclass} & {extent:.0f} & {resolution:.0f} & {quantity:.1f} & {uncertainty:.2f} & {gcps:.1f} & {dispersed:.2f} & {projection:.2f} & {metanoise:.2f} & {formatnoise:.2f} \\'.format(errclass=f['errclass'],
                                                                                                                                  extent=f['extent'],
                                                                                                                                  resolution=f['resolution'],
                                                                                                                                  quantity=f['quantity'],
                                                                                                                                  uncertainty=f['uncertainty'],
                                                                                                                                  gcps=f['gcps'] or float('nan'),
                                                                                                                                  dispersed=f['dispersed'],
                                                                                                                                  projection=f['projection'],
                                                                                                                                  metanoise=f['metanoise'],
                                                                                                                                  formatnoise=f['formatnoise'],
                                                                                                                                  )
    print latexrow
    
# table of regression results
# ...
    
    
    
    
    
    
fsadfas


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
######################## OLD #################
# PROSPECTUS
sns.set_style("darkgrid")

# simulation stats
def table():
    # 9 different centers
    import classypie as cp
    classfunc = lambda v: cp.find_class(v, [0, 2.5, 15, 50, 20000])[0] if not math.isnan(v) else None
    stats.compute('errclass', lambda f: classfunc(f['error_auto']['avg']/1000.0) if f['error_auto'] else None)
    agg = stats.select(lambda f: f['opts']).aggregate(key=['errclass'],
                    fieldmapping=[('count',lambda f: f['sim_id'],'count'),
                                 ]
                   )
    for f in agg:
        print f.row
        
table()
    
# error sources table
def table():
    import classypie as cp
    classfunc = lambda v: cp.find_class(v, [0, 2.5, 15, 50, 20000])[0] if not math.isnan(v) else None
    stats.compute('errclass', lambda f: classfunc(f['error_auto']['avg']/1000.0) if f['error_auto'] else None)
    agg = stats.select(lambda f: f['opts']).aggregate(key=['errclass'],
                    fieldmapping=[('count',lambda f: f['sim_id'],'count'),
                                  ('extent',lambda f: f['opts']['regionopts']['extent']*100,'avg'),
                                 ('resolution',lambda f: f['opts']['resolution'],'avg'),
                                  ('quantity',lambda f: f['opts']['placeopts']['quantity'],'avg'),
                                  ('uncertainty',lambda f: f['opts']['placeopts']['uncertainty']*100,'avg'),
                                  ('gcps',lambda f: f['error_auto']['labels_used'] * float(f['opts']['placeopts']['quantity']) if f['error_auto'] else None,'avg'),
                                 ]
                   )
    for f in agg:
        yield f
         
for f in table():
    latexrow = '{errclass} & {extent:.0f} & {resolution:.0f} & {quantity:.1f} & {uncertainty:.2f} & {gcps:.1f} \\'.format(errclass=f['errclass'],
                                                                                                                                  extent=f['extent'],
                                                                                                                                  resolution=f['resolution'],
                                                                                                                                  quantity=f['quantity'],
                                                                                                                                  uncertainty=f['uncertainty'],
                                                                                                                                  gcps=f['gcps'],
                                                                                                                                  )
    print latexrow

# error source regression
import numpy as np
import statsmodels.api as sm

x = np.array([[f['opts']['regionopts']['extent'] if f['opts'] else float('nan'),
               f['opts']['resolution'] if f['opts'] else float('nan'),
               f['opts']['placeopts']['quantity'] if f['opts'] else float('nan'),
               f['opts']['placeopts']['uncertainty'] if f['opts'] else float('nan'),
               f['error_auto']['labels_used'] * float(f['opts']['placeopts']['quantity']) if f['error_auto'] else float('nan')]
              for f in stats])
y = np.array([f['error_auto']['avg'] if f['error_auto'] else float('nan')
              for f in stats])

x = sm.add_constant(x)
results = sm.OLS(y, x, missing='drop').fit()
print results.summary().as_latex()
  
# graph autogeoref errors
def graph():
    vals = [f['error_auto']['avg'] for f in stats if f['error_auto']]
    vals = [v/1000.0 for v in vals if not math.isnan(v)]
    vals = [v for v in vals if 0 <= v <= 60]
    g = sns.distplot(vals, bins=60, kde=False, norm_hist=False, hist_kws=dict(alpha=1))
    plt.xlabel('Mean Absolute Error (Km)')
    plt.ylabel('Georeferenced Maps')
    return g

graph()
  
#sns.kdeplot(vals, shade=True)






fdsaf





# test number of georeferences actually attempted
len(set(fil for fil in os.listdir('maps') if fil.endswith('auto_log.txt')))
# test number of georeferences that didn't complete
len(set(fil for fil in os.listdir('maps') if fil.endswith('auto_log.txt') and 'total runtime: ' not in open('maps/'+fil).read()))
# test number of georeferences that raised error
len(set(fil for fil in os.listdir('maps') if fil.endswith('auto_log.txt') and 'total runtime: ' not in open('maps/'+fil).read() and 'Error:' in open('maps/'+fil).read()))
# list errors
geterror = lambda text: text[text.find('Error:'):text.find('\n', text.find('Error:'))]
errors = [geterror(open('maps/'+fil).read()) for fil in os.listdir('maps') if fil.endswith('auto_log.txt') and 'total runtime: ' not in open('maps/'+fil).read() and 'Error:' in open('maps/'+fil).read()]
set(errors)
# test number of georeferences that timed out
len(set(fil for fil in os.listdir('maps') if fil.endswith('auto_log.txt') and 'total runtime: ' not in open('maps/'+fil).read() and 'Error:' not in open('maps/'+fil).read()))












####################
# OLD BELOW
####################
    
# EXACT

# avg surface
vals = [f['error_exact']['avg'] for f in stats if f['error_exact']]
plt.hist(vals, bins=100) #, range=(0,10000))
plt.title('Average errors')
plt.show()

# avg vs extent
xys = [(f['opts']['regionopts']['extent'],f['error_exact']['avg']) for f in stats if f['error_exact'] and 'avg' in f['error_exact']]
xs,ys = zip(*xys)
plt.scatter(xs, ys)
plt.xlabel('Extent')
plt.ylabel('Average errors')
plt.show()

# rmse gcps
vals = [f['error_exact']['rmse'] for f in stats if f['error_exact'] and 'rmse' in f['error_exact']] 
plt.hist(vals, bins=100) #, range=(0,10000))
plt.title('RMSE')
plt.show()

# rmse vs extent
xys = [(f['opts']['regionopts']['extent'],f['error_exact']['rmse']) for f in stats if f['error_exact'] and 'rmse' in f['error_exact']]
xs,ys = zip(*xys)
plt.scatter(xs, ys)
plt.xlabel('Extent')
plt.ylabel('RMSE')
plt.show()



# AUTO

# avg surface
vals = [f['error_auto']['avg'] for f in stats if f['error_auto']]
vals = [v for v in vals if not math.isnan(v)]
plt.hist(vals, bins=100, range=(0,60000))
plt.title('Average errors')
plt.show()

# avg vs extent
xys = [(f['opts']['regionopts']['extent'],f['error_auto']['avg']) for f in stats if f['error_auto'] and 'avg' in f['error_auto']]
xys = [(x,y) for x,y in xys if not math.isnan(y)]
xs,ys = zip(*xys)
plt.scatter(xs, ys)
plt.xlabel('Extent')
plt.ylabel('Average errors')
plt.show()



