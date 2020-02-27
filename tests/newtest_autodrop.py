
print 'importing libs'

from automap.main import automap

db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"

print 'beginning'

#automap('testmaps/brazil_pol_1981.gif', textcolor=None, warp_order=None, db=db, debug=True)
automap('testmaps/burkina.jpg', textcolor=None, warp_order=None, db=db, debug=True)



