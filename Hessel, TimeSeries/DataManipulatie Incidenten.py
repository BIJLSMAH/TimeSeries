#%% Inlezen gegevens incidenten
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\PythonData\brongegevens productie\Alle Incidenten UTF-8.csv',delimiter=";" )

# df.info()
# df.describe()
# df.head(30)
# df.tail(30)
#%% Maak aanvullende attributen aan voor maand aangemeld en maand afgemeld.

# Gebruik pandas to_datetime om er een datumtijd object van te maken.
# Vergeet niet de dayfirst = True omdat anders maanden en dagen kunnen worden omgedraaid.
df['Datum Aangemeld'] = pd.to_datetime(df['Datum Aangemeld'], dayfirst=True)
df['Datum Afgemeld'] = pd.to_datetime(df['Datum Afgemeld'], dayfirst=True)

# Bereken de jjjjmm voor Aangemeld en Afgemeld
df['mndAangemeld'] = df['Datum Aangemeld'].dt.year * 100 + df['Datum Aangemeld'].dt.month
df['mndAfgemeld'] = df['Datum Afgemeld'].dt.year * 100 + df['Datum Afgemeld'].dt.month

# Vervang de nullwaarden met defaultwaarden om fouten te voorkomen
df['mndAangemeld'].fillna(200001, inplace=True)
df['mndAfgemeld'].fillna(203001, inplace=True)

# Door de formule kan een decimaal achter ontstaan. Maak hiervan een long
df['mndAangemeld'] = df['mndAangemeld'].astype(np.int64)
df['mndAfgemeld'] = df['mndAfgemeld'].astype(np.int64)
#%% Maak nu de subtotalen per maand
# Maak eerst een lijst met maanden van/tm
from datetime import datetime

def monthlist(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1 ,1))
    return mlist

# Doorloop de maanden en verzamel maandgegevens over de totalen
dates=["2013-06-01", "2020-01-01"]
mndLijst = monthlist(dates)

# Maak er een dataframe van die kan worden aangevuld

dfOpen = pd.DataFrame(columns =['maand', 'datum', 'open']) 

nieuw_record = pd.Series({'maand': 0, 'datum': 0, 'open': 0 })
for maand in mndLijst:
    jaarmaand = (maand.year * 100) + maand.month
    nieuw_record['maand'] = jaarmaand
    nieuw_record['datum']= str(maand)
    nieuw_record['open'] = len(df[(df.mndAangemeld <= jaarmaand) & (df.mndAfgemeld >= jaarmaand)])
    dfOpen = dfOpen.append(nieuw_record, ignore_index=True)

dfOpen.set_index('datum')
#%% plot de resultaten en sla ze op voor een vervolgbewerking

import matplotlib.pyplot as plt
import numpy as np
import pickle

x = dfOpen['datum']
y = dfOpen['open']
plt.plot(x,y)

pickle.dump(dfOpen, open('TSeries','ab'))
#%% Alleen de storingen filteren

dfstoring = df[df['Soort Incident']=='Storing']

