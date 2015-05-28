import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bioservices import KEGG, KEGGParser
from pandas import DataFrame, Series, read_csv

# Set-up bioservices
bio_kegg, bio_kegg_p = KEGG(cache=True), KEGGParser()
bio_kegg.organism = 'sce'
print '[INFO] Kegg Bioservice configured'

# Get organism reactions information
k_reactions = {r: bio_kegg.get(r) for r in bio_kegg.reactionIds}
print '[INFO] Kegg reactions extracted: ', len(k_reactions)