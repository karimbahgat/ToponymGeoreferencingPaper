
print 'importing libs'

from automap.main import automap

db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"

print 'beginning'

automap('testmaps/brazil_pol_1981.gif', textcolor=None, warp_order=2, db=db, debug=True)



