
print('importing libs')

from automap.main import automap

db = r"C:\Users\kimok\Desktop\gazetteers\gazetteers.db"

print('beginning')
#fil = 'testmaps/china_pol96.jpg'
fil = r'C:\Users\kimok\Desktop\0f97747b4f88.jpg'
res = automap(fil, textcolor=(0,0,0), db=db, outpath=False, debug=True)

print(res.keys())
