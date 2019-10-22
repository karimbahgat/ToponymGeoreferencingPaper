
import pythongis as pg
import matplotlib.pyplot as plt
import json
import os

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
plt.hist(vals, bins=100, range=(0,150000))
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



