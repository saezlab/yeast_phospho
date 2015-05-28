import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series, read_csv

wd = '/Users/emanuel/Projects/projects/yeast_phospho/'

tf_df = read_csv(wd + 'tables/tf_enrichment_df.tab', sep='\t')