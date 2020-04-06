
from mapsearch import MapDB

import PIL, PIL.Image
import os
import urllib
import io


# ACTUALLY, maybe dont use mapsearch
# just use the old approach of downloading to folder
# then processing each


# params
SCRAPE = False
ROOT = "https://legacy.lib.utexas.edu/maps/thailand.html" # should be part of the perry cestanada website

# init
db = MapDB('testdata/test_web.db')

if SCRAPE:
    # define images to process
    raw = urllib.urlopen(ROOT).read()
    elems = raw.replace('>', '<').split('<')

    # loop image links, process, and insert
    for elem in elems:
        if elem.startswith('a href='):
            url = elem.replace('a href=', '').strip('"')

            if url.endswith(('.png','.jpg','.gif')):
                if not url.startswith('http'):
                    url = 'https://legacy.lib.utexas.edu/maps/' + url

                if db.get('select 1 from maps where link = ?', (url,) ):
                    # skip if already exists
                    continue

                print 'loading', url
                fobj = io.BytesIO(urllib.urlopen(url).read())
                img = PIL.Image.open(fobj)

                if img.size[0] > 3000:
                    continue

                print 'processing'
                db.process(url, img)



