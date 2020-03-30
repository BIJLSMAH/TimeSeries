# DDS Servicedesk en Incidentmanagement 2020

Project "Data Science voor de Servicedesk en Incidentmanagement" vanuit DICTU Data Services. 

## Gegevens
Ontwikkelaars: Hessel Bijlsma, Alexander Harms  
Begin project: 28 januari 2020

## Data
In de programmatuur wordt uitgegaan van de data-mappen './data/raw' en 
'./data/processed' vanaf de root van dit project. In de map 'raw' staan de 
originele datasets die aangeleverd zijn; in de map 'processed' staan alle
afgeleide datasets. In de map 'processed' kan een verdere onderverdeling gemaakt worden
per project-onderdeel als dit het overzicht ten goede komt.

In deze sectie staat een korte opsomming van gegevens over de datasets.
### Raw
#### 20190711 - Incidenten Export - PROD.csv
Algemene export van de incidenten uit Topdesk.   
Datum export: 11 juli 2019  
Verkregen via: Jhonny Nieland  
Omgeving: Productie  
Incidenten in export van ???? tot ????  
Deze dataset heeft zowel eerste als tweedelijnsincidenten.  

#### 20190828 - Historie Incidenten - ACC.csv
Dataset met per incident de statuswijzigingen, voor process mining.  
Datum export: 28 augustus 2019  
Verkregen via: Gert-Jan Kamies  
Omgeving: Acceptatie  
Incidenten in export van 2 januari 2018 tot 25 juli 2019.  
Deze dataset heeft alleen tweedelijnsincidenten.  

#### 20200305 - Incidenten Tijdreeksanalyse - PROD.csv
Dataset met incidenten en hun aanmeld- en afmelddatum en het type incident.
Deze dataset bestaat uit de incidenten van 2 januari 2013 tot 31 december 2019.
Verkregen via: Marieke Paardekoper  
Omgeving: Productie

### Processed
#### Tijdreeks_Incidenten.csv
Deze dataset bestaat uit het eindproduct van ./Tijdreeksanalyse/conversie.py 
uitgevoerd op 20200305 - Incidenten Tijdreeksanalyse - PROD.csv.

## Gebruik
In deze sectie wordt beschreven hoe de scripts gebruikt moeten worden. 

### Tijdreeksanalyse
Scripts voor tijdreeksanalyse vindt men in de map ./Tijdreeksanalyse.  
Met conversie.py kan de export uit Topdesk, zie 20200305 - Incidenten Tijdreeksanlayse - 
PROD.csv, omgezet worden in een dataset waar per rij een dag beschreven wordt 
en per kolom een type incident inclusief het totaal. Per type incident is een kolom met het 
aantal aangemelde, het aantal afgemelde en het aantal openstaande incidenten op die dag.  

Met analyse.py kan men verschillende modellen testen over een test-periode met de 'rolling
 horizon'-techniek. De modelilen kunnen in een apart Python-bestand gedefinieerd worden.  

### Software
Hier vindt men informatie over de benodigde software om de programmatuur te 
laten draaien.
