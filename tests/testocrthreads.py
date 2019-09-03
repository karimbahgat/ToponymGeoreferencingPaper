
import PIL, PIL.Image

import pytesseract

import threading
from time import time
from random import uniform

def detect_data(im, bbox=None):
    if bbox:
        im = im.crop(bbox)
    data = pytesseract.image_to_data(im, lang='eng+fra', config='--psm 11') # +equ
    drows = [[v for v in row.split('\t')] for row in data.split('\n')]
    dfields = drows.pop(0)
    drows = [dict(zip(dfields,row)) for row in drows]
    print '-------------\n%s\n%s\n' % (len(drows), '\n'.join([r['text'] for r in drows if r.get('text')]))
    return drows

im = PIL.Image.open('testmaps/burkina.jpg')
n = 2
#im = PIL.Image.open('testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg')
#n = 5

im = im.convert('L')
#im.show()

# all in one
if 1:
    t = time()
    detect_data(im)
    print time()-t

# chained threaded
import pyagg
c = pyagg.canvas.from_image(im)
tw = im.size[0] / n
th = im.size[1] / n
threads = []
t = time()
for px in range(0, im.size[0], tw):
    for py in range(0, im.size[0], th):
        bbox = [px, py, px+tw, py+th]
        if uniform(0, 1) > 0.2: continue
        print bbox
        c.draw_box(bbox=bbox, fillcolor=None, outlinecolor=(0,255,0), outlinewidth=1)
        thr = threading.Thread(target=lambda: detect_data(im, bbox=bbox))
        thr.daemon = True
        thr.start()
        threads.append(thr)
c.get_image().convert('RGB').save('hmm.jpg')
while any((thr.is_alive() for thr in threads)):
    pass
##for thr in threads:
##    thr.join()
print time()-t

