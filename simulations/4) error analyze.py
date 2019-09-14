
import pythongis as pg
import matplotlib.pyplot as plt
import json

# - take all permutations
# - associate each with its avg error metric
# - use error metric to cluster combinations of parameter values/ranges that result in lots of error
# - thus can know which types of maps are more suitable, and which method params work best

stats = []

for i in range(250):
    print i
    try:
        raw = open('maps/sim_{}_error.json'.format(i)).read()
        dct = json.loads(raw)
        stats.append(dct)
    except:
        pass

avgs = [s['avg'] for s in stats]
print len(stats)
plt.hist(avgs, bins=20, range=(0,10000))
plt.show()


