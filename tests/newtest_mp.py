
import automap as mapfit
import os
#testim = 'txu-oclc-6654394-nb-30-4th-ed.jpg'
testim = 'burkina.jpg'
#testim = 'china_pol96.jpg'
#testim = 'brazil_land_1977.jpg'
#testim = 'france_admin91.jpg'
#testim = 'russia_autonomous92.jpg'
#testim = 'israel-and-palestine-travel-reference-map-[2]-1234-p.jpg'
#testim = 'korean_war_april_1951.jpg'
testim_root,ext = os.path.splitext(testim)

# first produce
db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"
if __name__ == '__main__':
    info = mapfit.automap('testmaps/{}'.format(testim), textcolor=None, warp=False, warp_order=None, db=db, debug=True, parallel=True, max_procs=8)



### image
##render = mapfit.debug.render_text_recognition('testmaps/{}'.format(testim),
##                                              'testmaps/{}_georeferenced.tif'.format(testim_root))
##render.save('testdebugimage.png')
##
### georef
##render = mapfit.debug.render_georeferencing('testmaps/{}_georeferenced.tif'.format(testim_root))
##render.save('testdebuggeoref.png')






