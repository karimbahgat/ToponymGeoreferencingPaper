
print('importing libs')

from automap import automap
#from automap.main import debug_warped

#db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"
db = r"P:\(Temp Backup)\gazetteer data\optim\gazetteers.db"

print('beginning')

# debug
##import pythongis as pg
##base = 'israel-and-palestine-travel-reference-map-[2]-1234-p'
##d = pg.VectorData('testmaps/%s_controlpoints.geojson' % base)
##controlpoints = [[f['origname'], (f['origx'],f['origy']), f['matchname'], (f['matchx'],f['matchy']), f['residual']]
##                 for f in d]
##debug_warped('testmaps/%s_georeferenced.tif' % base, 'hmm.png', controlpoints)
##fdfs

# works

#automap('testmaps/HandDrawnMap.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/old map sketch.jpg', textcolor=(90,90,90), colorthresh=25, textconf=60)
#fdsf

#automap('testmaps/satim.png', textcolor=(0,0,0), colorthresh=25, textconf=60)

#automap('testmaps/burkina.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, db=db, debug=True)
automap('testmaps/burkina_pol96.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, db=db, debug=True)
#automap('testmaps/ghana_rel_1983.jpg', db=db, source='best', debug=True)
#automap('testmaps/tunisia_pol_1990.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/belgium.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/nepal_pol90.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/cameroon_pol98.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/cameroon.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, max_residual=0.2)
#automap('testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)

#automap('testmaps/repcongo.png', textcolor=(120,120,120), colorthresh=25, textconf=60)
#automap('testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, bbox=[2000,2500,4000,4500])
#automap('testmaps/gmaps.png', textcolor=(80,80,80), colorthresh=25, textconf=60)
#automap('testmaps/brazil_pol_1981.gif', textcolor=(0,0,0), colorthresh=25, textconf=60, max_residual=0.4)
#automap('testmaps/china_pol96.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, db=db, debug=True, warp_order=3)#, max_residual=0.4)

#automap('testmaps/vietnam_pol92.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60)
#automap('testmaps/vietnam_admin92.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60)
#automap('testmaps/nigeria_econ_1979.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60, max_residual=0.5)
#automap('testmaps/nigeria_crops.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60, max_residual=0.5)
#automap('testmaps/nigeria_linguistic_1979.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60, max_residual=0.5)

#automap('testmaps/namibia_homelands_78.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/namibia_pol90.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)

#automap('testmaps/washington_baltimore.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60)



# difficult
#automap('testmaps/2113087.jpg', textcolor=(120,120,120), colorthresh=25, textconf=60, bbox=[1000,1000,2000,2000])
#automap('testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg', textcolor=(50,50,50), colorthresh=25, textconf=90, bbox=[3000,1500,4000,2500])
#automap('testmaps/ierland-toeristische-attracties-kaart.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60)
#automap('testmaps/brazil_pop_1977.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60, max_residual=0.6)
#automap('testmaps/brazil_land_1977.jpg', textcolor=(50,50,50), colorthresh=40, textconf=60)
#automap('testmaps/egypt_admn97.jpg', textcolor=(0,0,0), colorthresh=40, textconf=60)
#automap('testmaps/devils tower small.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, maxiter=3000, mintrials=1000)
#automap('testmaps/brazil_army_amazon_1999.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60, maxiter=30000, mintrials=1000, max_residual=0.5)
#automap('testmaps/egypt_pol_1979.jpg', textcolor=(0,0,0), colorthresh=40, textconf=30, db=db, warp_order=3)#, max_residual=0.005)
#automap('testmaps/CHHJ5246_Updated_areasOfConflict_Map_0217_V2.png', textcolor=(0,0,0), colorthresh=40, textconf=60)
#automap('testmaps/korean_war_april_1951.jpg', textcolor=(0,0,0), colorthresh=40, textconf=60)

#automap('testmaps/zaire_map.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/russia_autonomous92.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/algeria_rel79.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/france_admin91.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)



