
import pythongis as pg
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math

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
    out.fields = ['sim_id', 'opts', 'error_auto', 'error_exact']

    roots = set(('_'.join(fil.split('_')[:4]) for fil in os.listdir('maps')))

    for root in roots:
        print root

        vals = {'sim_id':root}

        opts_path = 'maps/{}_opts.json'.format(root)
        if os.path.lexists(opts_path):
            with open(opts_path) as fobj:
                opts = json.load(fobj)
                vals['opts'] = opts
                
    ##            for k,v in opts.items():
    ##                vals['opt_'.format(k)] = v

        erraut_path = 'maps/{}_georeferenced_auto_error.json'.format(root)
        if os.path.lexists(erraut_path):
            with open(erraut_path) as fobj:
                error_auto = json.load(fobj)
                vals['error_auto'] = error_auto

    ##            for k,v in error_auto.items():
    ##                vals['erraut_'.format(k)] = v

        errexa_path = 'maps/{}_georeferenced_exact_error.json'.format(root)
        if os.path.lexists(errexa_path):
            with open(errexa_path) as fobj:
                error_exact = json.load(fobj)
                vals['error_exact'] = error_exact

    ##            for k,v in error_exact.items():
    ##                vals['errexa_'.format(k)] = v

        print vals
        
        out.add_feature(vals, None)


    # encode dicts to json for saving
    for fl in out.fields:
        out.compute(fl, lambda f: json.dumps(f[fl]))

    # save
    out.save('analyze/stats.xls')





##############

stats = pg.VectorData('analyze/stats.xls')
for fl in stats.fields:
    stats.compute(fl, lambda f: json.loads(f[fl]))


    
    
    
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



