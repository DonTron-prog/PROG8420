import pandas as pd
import re
import numpy as np

GOC = pd.DataFrame(pd.read_csv('/home/donald/Dropbox/Energy_analysis/GOC/GOC.csv'))

sums = GOC.sum()
#chage series to dataframe
sums = (sums.to_frame())
sums = sums.rename(columns= {0:'Sum'})

length = GOC.count()
length = length.to_frame()
length = length.rename(columns={0: "Count"})

stats = pd.concat([sums, length], axis=1)

stats = stats.sort_values(by = ['Sum'], ascending=False)
print (stats.head(10))


#sumdf = pd.DataFrame(g, columns = ['Facility', 'Total Output'])
#print (sumdf)
