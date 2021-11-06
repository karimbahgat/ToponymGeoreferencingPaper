import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import math
import sqlite3




print(os.getcwd())
try:
    os.chdir('simulations')
except:
    pass




# Loop and create table
if True:
    out = []

    for fil in os.listdir('scrape'):
        #print fil
        if fil.endswith('.json'):
            with open('scrape/'+fil) as fobj:
                dct = json.load(fobj)
                dct['filename'] = fil
                del dct['thumbnail']
        
            out.append(dct)

print(len(out))

# drop weird maps from the automatic scraping process 
# (pure islands, no toponyms, non-maps)

dropfiles = ['0ad520bf73b8c8906cbe31ed5ba7a7726f75473c00c337cb5e94572f.json', '15869a2cabb66f3c2d70ffd0338e83d8e7e5c4060a94f1c3a0166062.json', '1c1bf2ff40bce027f1b486a6e01b185e77a26df2a41325ebf1c0a3f1.json', '6552b0728bf127a41b89ad089783989649dfb739eadce9826ff702b3.json', '69356071f9f0d04f3c07682eaef07e55f5bc45d4325336f5b8d1fb6d.json', '7644d9dcda6d3afa8ee16d1f693a863cb0608233c97f5c302c23bd02.json', '77fee64502c3b086389b5453fa40f7abcb80a41eee689307fb43e55c.json', '782d26f76e13d29bd83ec92b811adaee926c71e2e58c2e912913d309.json', '86b774d09c6da46ae3ef9c084f23a579d2b4407ec2bcc8555afa887f.json', '8708f351b77c9762948b962db3cf2d9e15d8ee8ba7edb04063cf53b1.json', '8b0030f673beb43a852efd71c1841552e8e0bf8e6bb09803a7e63812.json', '923df4a25a50caff6d7f68559a486540c792fca3f6d545c6e1f556ad.json', 'a17f7e8c39c24e2547cc04ee628341ac16ad07c58c28b66ac79fe6db.json', 'a6ee245417147630d7fe3cb9e1ae6d088ea1dff66a7187416f19aa10.json', 'b5328a7cf7e930b241eb5b1f7a7912772a1c395bfe68f99cbc3d2f6a.json', 'bc2005bfb0188d1ce4a645540091c22f7de8979e3dac5fd8118ec75c.json', 'bf27f8edce01451c6556c175de3350884d7f0fd24884949bf40c505a.json', 'bf68323a896b424d694e4144d9857047487ab390e33986270c5f6e4b.json', 'c70730f9ac2cebb546885e2ad63a4f12bee70a53b81e42bb937393d3.json', 'cec085d33fa142d1924bd9f7541c1deaf4006a5df310d9bf630306d1.json', 'd1159a0a2bffc69dc89033bda1179a981a7d362701c31bbb6185753f.json', 'd4c4e8bdde44903724fd89eeab5d6a6471b2e21fc41e655c1f0e175d.json', 'dd8e7601355c1c9e9b0bd09690a8f609f527a1804c7e607eae47d571.json', 'ddef35f92e98f8a7a98930f474dd2299d766f2295c8a6fb34c67fbcc.json', 'df63d87fef946a3bce9324dc654f2e9fbc08024344d5780d8b045990.json', 'e147db4d133485bc5d204f888e42b44022f52a710f44c82848bd5e4e.json', 'e59ccdc48b6ac60ec402fe1ed82ba9855e4551d327400e6eadacd4ca.json', 'e6de2462cabaf33f3693cfbf5f39de569d3ff8a07734b65c68fff4d6.json', 'e8149c004cfd63782350f0e16ca68b1733cd03219fc519a0633edc5f.json', 'eb16e46b1a823776c926cb9940ee5e5ad5cf50124356b725d58f3709.json', 'f1731fe1077567f5aab24613a24a146ed077f150762f84da3cdd5ea5.json', 'f8cb8c30e3fb5b6cf32217842330bbdd5a78d48de504a5e3f322b34c.json']
out = filter(lambda f: f['filename'] not in dropfiles,
            out)
out = list(out)

print(len(out))




######



from landsatxplore.api import API
username = 'kbahgat'
password = 'Murdockdog12'
landsat_api = API(username, password)

for meta in out:
    print(meta['url'])
    bbox = meta['georef']['bbox']
    scenes = landsat_api.search(
        dataset='landsat_8_c1',
        bbox=bbox,
        max_cloud_cover=10,
    )
    for scene in scenes:
        print(scene)
        # download
        #ee.download('LT51960471995178MPS00', output_dir='./data')
        # load
        #r = pg.RasterData(pth)
        #m.add_layer(r)
    
    fdasfas

