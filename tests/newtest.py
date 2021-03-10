
print 'importing libs'

from automap.main import automap

db = r"C:\Users\kimok\Desktop\BIGDATA\gazetteer data\optim\gazetteers.db"

print 'beginning'

#automap('testmaps/burkina.jpg', textcolor=(40,30,20), sample=False, db=db, debug=True)
#automap('testmaps/tunisia_pol_1990.jpg', textcolor=None, db=db)
#automap('testmaps/belgium.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/nepal_pol90.jpg', textcolor=(0,0,0), db=db)
#automap('testmaps/cameroon_pol98.jpg', textcolor=None, db=db, debug=True)
automap('testmaps/cameroon.jpg', textcolor=(0,0,0), db=db, debug=True)
#automap('testmaps/israel-and-palestine-travel-reference-map-[2]-1234-p.jpg', textcolor=None, colorthresh=25, sample=False, textconf=60, db=db)

#automap('testmaps/repcongo.png', textcolor=None, db=db) #(120,120,120))
#automap('testmaps/txu-pclmaps-oclc-22834566_k-2c.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, bbox=[2000,2500,4000,4500])
#automap('testmaps/gmaps.png', textcolor=(80,80,80), colorthresh=25, textconf=60)
#automap('testmaps/brazil_pol_1981.gif', textcolor=None, db=db, debug=True)
#automap('testmaps/china_pol96.jpg', textcolor=None, db=db, debug=True)

#automap('testmaps/vietnam_pol92.jpg', textcolor=(50,50,50), db=db, debug=True)
#automap('testmaps/vietnam_admin92.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60)
#automap('testmaps/nigeria_econ_1979.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60, max_residual=0.5)
#automap('testmaps/nigeria_crops.jpg', textcolor=None, db=db)
#automap('testmaps/nigeria_linguistic_1979.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60, max_residual=0.5)

#automap('testmaps/namibia_homelands_78.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/namibia_pol90.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)

#automap('testmaps/washington_baltimore.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60)



# difficult
#automap('testmaps/2113087.jpg', textcolor=(120,120,120), colorthresh=25, textconf=60, bbox=[1000,1000,2000,2000])
#automap('testmaps/txu-oclc-6654394-nb-30-4th-ed.jpg', textcolor=(50,50,50), colorthresh=25, textconf=90, bbox=[3000,1500,4000,2500])
#automap('testmaps/ierland-toeristische-attracties-kaart.jpg', textcolor=None, db=db)
#automap('testmaps/brazil_pop_1977.jpg', textcolor=(50,50,50), colorthresh=25, textconf=60, max_residual=0.6)
#automap('testmaps/brazil_land_1977.jpg', textcolor=None, db=db) #(50,50,50))
#automap('testmaps/egypt_admn97.jpg', textcolor=(0,0,0), colorthresh=40, textconf=60)
#automap('testmaps/devils tower small.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60, maxiter=3000, mintrials=1000)
#automap('testmaps/brazil_army_amazon_1999.jpg', textcolor=(50,50,50), colorthresh=35, textconf=60, maxiter=30000, mintrials=1000, max_residual=0.5)
#automap('testmaps/egypt_pol_1979.jpg', textcolor=(0,0,0), colorthresh=40, textconf=60, max_residual=0.005)
#automap('testmaps/CHHJ5246_Updated_areasOfConflict_Map_0217_V2.png', textcolor=(0,0,0), colorthresh=40, textconf=60)
#automap('testmaps/korean_war_april_1951.jpg', textcolor=(0,0,0), colorthresh=40, textconf=60)

#automap('testmaps/zaire_map.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/russia_autonomous92.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/algeria_rel79.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)
#automap('testmaps/france_admin91.jpg', textcolor=(0,0,0), colorthresh=25, textconf=60)



