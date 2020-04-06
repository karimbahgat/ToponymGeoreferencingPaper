
from mapsearch import MapDB


# ACTUALLY, maybe dont use mapsearch
# just use the old approach of downloading to folder
# then processing each


db = MapDB('testdata/test_web.db')

root = 'http://legacy.lib.utexas.edu/maps/onc/europe-asia-africa-index.html'

for link in iter_links():
    print 'loading', link
    
    fobj = io.BytesIO(urllib.urlopen(link).read())
    img = PIL.Image.open(fobj)
    
    db.process(url, img)




