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

# all renderings
print stats

# exclude maps with nonlegible texts (jpg at 1000 pixel resolution)
stats_all = stats
stats = stats.select(lambda f: not f['sim_id'].endswith('_6'))
print stats

# scenes
scenes = set([f['sim_id'][:6] for f in stats])
print len(scenes)
  
  
  
    
  
    
    
    

    
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
    fig.subplots_adjust(bottom=0.1)
    fig.savefig('analyze/figures/error_km_by_extent.png')
    
plot('max', range=(0,5000))

examps = [f for f in stats if f['accuracy'] and 800 < f['accuracy']['rmse_georeferenced']['geographic'] < 1200]
for f in examps[0:1]:
    print f['accuracy']['rmse_georeferenced']
    display(map_overlay(f['sim_id']))

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
def subplot(ax, sim_id, subtitle):
    # map overlay
    im = map_overlay(sim_id)
    ax.set_title(subtitle)
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(im)

# fig
fig,axes = plt.subplots(3, 2, 
                        figsize=(6,10))
# out of wack
examps = [f for f in stats if f['accuracy'] and 2 < f['accuracy']['max_georeferenced']['percent'] < 3]
for f in examps[8:9]:
    print f['accuracy']['max_georeferenced']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_wack.png')
    #plot(f['sim_id'], 'wack')
    #subplot(axes[2,0], f['sim_id'], 'not usable\n'+'>100%')
# approx but unusable
examps = [f for f in stats if f['accuracy'] and 0.5 < f['accuracy']['max_georeferenced']['percent'] < 0.6]
for f in examps[2:3]:
    print f['accuracy']['max_georeferenced']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_needsadj.png')
    #plot(f['sim_id'], 'approx_bad')
    #subplot(axes[1,1], f['sim_id'], 'approximate, needs fixing\n'+'>20%')
# approx
examps = [f for f in stats if f['accuracy'] and 0.1 < f['accuracy']['max_georeferenced']['percent'] < 0.12]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_approx.png')
    #plot(f['sim_id'], 'approx')
    #subplot(axes[1,0], f['sim_id'], 'approximate\n'+'>5%')
# reasonable
examps = [f for f in stats if f['accuracy'] and 0.01 < f['accuracy']['max_georeferenced']['percent'] < 0.05]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
    maps = map_overlay(f['sim_id'])
    display(maps)
    maps.save('analyze/figures/mapqual_reasonable.png')
    #plot(f['sim_id'], 'reasonable')
    #subplot(axes[0,1], f['sim_id'], 'reasonable\n'+'>1%')
# excellent
examps = [f for f in stats if f['accuracy'] and f['accuracy']['max_georeferenced']['percent'] < 0.01]
for f in examps[:1]:
    print f['accuracy']['max_georeferenced']
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
        if f['accuracy']:
            v = f['accuracy'][error_type+'_georeferenced']['percent']
            return cp.find_class(v, [0, 0.01, 0.05, 0.2, 1, 110000000])[0] if not math.isnan(v) else None
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
        if f['accuracy']:
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



# boxplots of good quality maps, adjusting one parameter at a time
# define params
resolutions = [3000, 2000, 1000]
extents = [50, 10, 1, 0.1] # ca 5000km, 1000km, 100km, and 10km
quantities = [80, 40, 20, 10]
uncertainties = [0, 0.01, 0.1, 0.5] # ca 0km, 1km, 10km, and 50km
distributions = ['dispersed', 'random'] # IMPROVE W NUMERIC
projections = [None, # lat/lon
               '+proj=robin +datum=WGS84 +ellps=WGS84 +a=6378137.0 +rf=298.257223563 +pm=0 +lon_0=0 +x_0=0 +y_0=0 +units=m +axis=enu +no_defs', #'+init=ESRI:54030', # Robinson
               ] 
datas = [False, 
         True] 
metas = [False, # nothing
         True, # text noise + meta boxes (arealabels + title + legend)
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
              projections=lambda f: f['rendopts']['projection'],
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
        vals = [v*100 for v in vals]
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

boxplot_stats(axes[7], 'distributions', distributions, )

boxplot_stats(axes[8], 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05,
                   hspace=0.30) #wspace=0.05)
#fig.delaxes(axes[-1])
#fig.suptitle('Effect of each parameter \n while keeping all others at baseline')
fig.savefig('analyze/figures/errors_baseline_boxplots.png')



# same boxplots but broken down by extent
fig,axes = plt.subplots(8, 4, 
                        #sharex=True, sharey=True,
                        figsize=(8,10))

j = 0

ax = axes[j,0]
ax.set_ylabel('max error (%)')
ax.annotate('img\nResolution', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    axes[0,i].set_title('extent = {} km'.format(extents[i]*100))
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
ax.annotate('toponym\nDispersed', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
for i in range(4):
    boxplot_stats(axes[j,i], 'distributions', distributions, xlabel=False, extents=extents[i])

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
        labelvals = [v*100 for v in vals]
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

outcomeplot_stats(axes[7], 'distributions', distributions, )

outcomeplot_stats(axes[8], 'uncertainties', uncertainties, )

fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15,
                   hspace=0.4) #wspace=0.05)
#fig.delaxes(axes[-1])
#fig.suptitle('Effect of each parameter \n while keeping all others at baseline')
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, frameon=False,
          loc='lower center')
fig.savefig('analyze/figures/outcomes_baseline_barplots.png')
  
  
###
# same barplots but broken down by extent
fig,axes = plt.subplots(4, 4, 
                        sharey=True, #sharex=True, sharey=True,
                        figsize=(8,5))

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

j += 1
ax = axes[j,0]
ax.set_ylabel('% of maps')
ax.annotate('num\nToponyms', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 12, 0),
                  xycoords=ax.yaxis.label, textcoords='offset points',
                  ha='center', va='center', rotation='vertical')
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


    
    
# table of error sources
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
                #distribution
                (1 if f['rendopts']['placeopts']['distribution'] == 'dispersed' else 0) if f['rendopts'] else float('nan'),
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
xnames = ['Constant', 'mapExtent', 'imgResolution', 'numToponyms', 'toponymUncertainty', 'toponymDispersed', 'numControlPoints', 'mapProjection', 'dataNoise', 'metaNoise', 'pixelNoise']
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
                #distribution
                (1 if f['rendopts']['placeopts']['distribution'] == 'dispersed' else 0) if f['rendopts'] else float('nan'),
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
for i in range(x.shape[1]):
    # min-max normalize predictors
    v = x[:,i][~np.isnan(x[:,i])]
    x[:,i] = (x[:,i] - v.min()) / (v.max() - v.min())
xnames = ['Constant', 'mapExtent', 'imgResolution', 'numToponyms', 'toponymUncertainty', 'toponymDispersed', 'mapProjection', 'dataNoise', 'metaNoise', 'pixelNoise']
y = np.array([f['accuracy']['max_georeferenced']['percent'] if f['accuracy'] else 999999999999999
              for f in stats])
y[y<0.05] = 1
y[(0.05<y) & (y<1)] = float('nan')
y[y>1] = 0
ynames = 'Success'

x = sm.add_constant(x)
results = sm.Logit(y, x, missing='drop').fit()
print results.summary(yname=ynames, xname=xnames).as_latex()
    
    
    
