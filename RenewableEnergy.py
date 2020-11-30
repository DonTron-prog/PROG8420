import pandas as pd
import re
import numpy as np

F1 = pd.DataFrame(pd.read_excel('/home/donald/Dropbox/Energy_analysis/Renewable_projects/rea.xlsx'))

#wind mills

wind = F1[F1["Project Type"] == "Wind"].copy()


pd_slice = wind.loc[:, ["Project Description"]]
f = []
for i in pd_slice["Project Description"]:
    f2 = re.findall(r'\d+\.*\d*', i)
    f.append(f2)
    #f = f[0].replace("'", "")
    #k = f[1].replace("'", "")
    #return f
print(f)
terb = []
wats = []
for i in range(0, len(f)):
    f1 = (f[i][0])
    terb.append(f[i][0])
    wats.append(f[i][1])

wind.insert(11, "MegaWatts", wats, True)
wind.insert(12, "Turbines", terb, True)

print(wind)



#solar

solar = F1[F1["Project Type"] == "Solar"].copy()


pd_solar_slice = solar.loc[:, ["Project Description"]]
f = []
for i in pd_solar_slice["Project Description"]:
    f2 = re.findall(r'\d+\.*\d*', i)
    f.append(f2)
    #f = f[0].replace("'", "")
    #k = f[1].replace("'", "")
    #return f

'''
wats = []
for i in range(0, len(f)):
    f1 = (f[i][0])
    terb.append(f[i][0])
    wats.append(f[i][1])
'''
terb = []
wats = []
for i in range(0, len(f)):
    terb.append(0)
    wats.append(f[i][0])



solar.insert(11, "MegaWatts", wats, True)
solar.insert(12, "Turbines", terb, True)

print (solar)


#bioenergy
bioenergy = F1[F1["Project Type"] == "Bioenergy"].copy()


pd_bio_slice = bioenergy.loc[:, ["Project Description"]]
f = []
for i in pd_bio_slice["Project Description"]:
    f2 = re.findall(r'\d+\.*\d*', i)
    f.append(f2)
    #f = f[0].replace("'", "")
    #k = f[1].replace("'", "")
    #return f

terb = []
wats = []
for i in range(0, len(f)):
    terb.append(0)
    wats.append(f[i][0])



bioenergy.insert(11, "MegaWatts", wats, True)
bioenergy.insert(12, "Turbines", terb, True)
print (bioenergy)

rea = wind.append([solar, bioenergy])

rea.to_csv('/home/donald/Dropbox/Energy_analysis/Renewable_projects/rea_organized.csv')
