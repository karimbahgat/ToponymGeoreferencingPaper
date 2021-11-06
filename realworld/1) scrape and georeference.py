
from urllib2 import urlopen
from PIL import Image
import json
import os
import io
import multiprocessing as mp
import base64
import automap as mapfit



print(os.getcwd())
try:
    os.chdir('realworld')
except:
    pass



MAXPROCS = 10



def iterurls():
    # custom african countries
    countries = '''Algeria
Angola
Benin
Botswana
Burkina Faso
Burundi
Cameroon
Cape Verde
Central African Republic
Chad
Comoros
Congo Republic
Zaire
Djibouti
Egypt
Equatorial Guinea
Eritrea
Ethiopia
Gabon
Gambia
Ghana
Guinea
Guinea-Bissau
Cote d'Ivoire
Kenya
Lesotho
Liberia
Libya
Madagascar
Malawi
Mali
Mauritania
Mauritius
Morocco
Mozambique
Namibia
Niger
Nigeria
Rwanda
Sao Tome
Senegal
Seychelles
Sierra Leone
Somalia
South Africa
South Sudan
Sudan
Swaziland
Tanzania
Togo
Tunisia
Uganda
Western Sahara
Zambia
Zimbabwe'''.split('\n')
    for country in countries:
        country = country.lower().replace(' ','_').replace('-','_')
        url = 'https://legacy.lib.utexas.edu/maps/{}.html'.format(country)
        yield country,url
        
def iterurls():
    # all countries listed on perry website
    index_url = 'https://legacy.lib.utexas.edu/maps/map_sites/country_sites.html'

    # parse website html
    raw = urlopen(index_url).read()
    raw = str(raw)
    elems = raw.replace('>', '<').split('<')

    # loop and identify country links
    urls = []
    for i,elem in enumerate(elems):
        if elem.startswith('a href='):
            nxt = elems[i+1]
            if nxt.endswith('(University of Texas at Austin Map Collection)'):
                country = nxt.split('(')[0].strip()
                # make relative links to absolute
                url = elem.replace('a href=', '').strip('"')
                if not url.startswith('http'):
                    url = 'https://legacy.lib.utexas.edu'+url
                yield country, url
                
def scrape_url(root_url, limit=2):
    # find img urls
    root_url_dir = os.path.split(root_url)[0]

    # parse website html
    raw = urlopen(root_url).read()
    raw = str(raw)
    elems = raw.replace('>', '<').split('<')
    
    # start looking right after 'Country Maps' elem
    if 'Country Maps' in elems:
        start = elems.index('Country Maps')
        elems = elems[start:]

    # loop and identify image links
    urls = []
    for elem in elems:
        if elem.startswith('a href='):
            url = elem.replace('a href=', '').strip('"')

            if url.endswith(('.png','.jpg','.gif','.tif')):
                # make relative links to absolute
                if not url.startswith('http'):
                    url = root_url_dir.strip('/') + '/' + url.strip('/')

                #print(url)

                # get filename
                filename = os.path.split(url)[-1]
                urls.append((filename,url))
                
                if len(urls) >= limit:
                    break
  
    # begin scrape
    for filename,url in urls:
        print(url)
            
        # add new
        print('adding')
        ###map_process(url)
        proc = pool.apply_async(map_process, (url,))
        pool_results.append(proc)
                
def map_process(url):
    # download image
    fobj = io.BytesIO(urlopen(url).read())
    img = Image.open(fobj)
    print(img)
    
    # add to db
    w,h = img.size
    mapp = dict(url=url, width=w, height=h)

    # create thumbnail
    max_size = 150
    longest = max(img.size)
    scale = max_size / float(longest)
    size = img.size[0] * scale, img.size[1] * scale
    thumb = img.copy()
    thumb.thumbnail(size)
    print(thumb)
    # encode to png bytestring
    fobj = io.BytesIO()
    thumb.save(fobj, "PNG")
    raw = base64.b64encode(fobj.getvalue())
    mapp['thumbnail'] = raw

    # calc short output filename
    import hashlib
    filename = hashlib.sha224(url).hexdigest() 
    
    # save the image
    img.save('scrape/'+filename+'.png')
    
    # georeference map? 
    res = map_georef(img)
    mapp['georef'] = res

    # save
    print('tosave',mapp.keys())
    with open('scrape/'+filename+'.json', 'w') as fobj:
        json.dump(mapp, fobj)

def map_georef(img): #url):
    # load img
    #fobj = io.BytesIO(urlopen(url).read())
    #img = Image.open(fobj)
    print(img)
  
    # set params
    params = dict(db='../data/gazetteers.db', 
                  source='best', 
                  textcolor=(0,0,0),
                  warp=False)

    # run
    res = mapfit.automap(img, **params)

    return res



if __name__ == '__main__':
    pool = mp.Pool(processes=MAXPROCS)
    pool_results = []
    for country,url in iterurls():
        print(country,url)
        scrape_url(url)

    # wait for results
    print('total tasks',len(pool_results))
    remaining = list(pool_results)
    prev = remaining
    while remaining:
        remaining = [proc for proc in remaining if not proc.ready()]
        if len(remaining) != len(prev):
            print('remaining tasks',len(remaining))
            prev = remaining
            


        
