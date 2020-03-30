# generateMeta.py
# Genereert 'metadata' van de databestanden in de /data map.
# Dit script gaat door alle CSV-bestanden in de /data map heen geeft per 
# bestand.csv een meta-bestand.txt terug met daarin basale informatie over de 
# dataset.
# - Aantal kolommen en rijen
# - Kolomnamen
import os
import pandas as pd

if not os.path.isdir("./data"): 
    print("Geen map /data. Script sluit nu af.")
    exit()

# Loop recursief door alle mappen heen en maak het meta-data bestand.
# Getting the current work directory (cwd)
thisdir = os.getcwd()
# r = root, d = directories, f = files
for r, d, f in os.walk(thisdir):
    for filename in f:
        if ".csv" in filename:
            print(filename)
            filepath = os.path.join(r, filename)
            # Door sep = None wordt er door csv.Sniffer gekeken welk
            # scheidingsteken er gebruikt wordt
            dataset = pd.read_csv(filepath, sep = None, engine = "python")
            rows = dataset.shape[0]
            columns = dataset.shape[1]
            column_names = dataset.columns.values

            meta_filename = "meta-" + filename[:-4] + ".txt"
            meta_filepath = os.path.join(r, meta_filename)
            with open(meta_filepath, 'w') as meta_file:
                meta_file.write("Kolommen: %d\n" % (columns))
                meta_file.write("Rijen: %d\n" % (rows))
                meta_file.write("Kolomnamen:\n")
                for name in column_names:
                    meta_file.write(name + "\n")
                


