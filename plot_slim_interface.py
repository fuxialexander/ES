#%%
import pandas as pd
import numpy as np
import argparse
import gzip
from locale import normalize
import os
import sys
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib import get_cachedir; print(get_cachedir()) 
plt.style.use('~/manuscript.mplstyle')
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.ndimage import gaussian_filter1d
from statannot import add_stat_annotation
from tqdm import tqdm
#%%

# %%
lddt = dict()
with open('plddt/9606.pLDDT.tdt', 'r') as f:
    for line in f:
        id, score = line.strip().split('\t')
        lddt[id] = np.array(score.split(",")).astype(float)
# %%
slim = pd.read_csv("./slim_df.csv")
interface = pd.read_csv("./interface_df.csv")
interface = pd.read_csv("./H_sapiens_interfacesHQ.txt", sep='\t')
genename_to_uniprot = pd.read_csv(
        "./uniprot_to_genename.txt", sep='\t').set_index('To').to_dict()['From']
uniprot_to_gene = pd.read_csv(
        "./uniprot_to_genename.txt", sep='\t').set_index('From').to_dict()['To']
slim['gene']=slim.uid.apply(lambda x: uniprot_to_gene[x] if x in uniprot_to_gene else np.nan)
slim=slim.dropna()
def comma_to_list(str):
        return str.strip('[]').split(',')

def dash_to_list(str):
    if '-' not in str:
        if len(str)==0:
            return np.nan
        else:
            return int(str)
    else:
        start, end = str.split('-')
        start = int(start)
        end = int(end)
        return list(range(start, end+1))
p2 = interface.set_index('P2').P2_IRES.apply(comma_to_list).explode().apply(dash_to_list).explode().dropna()
p1 = interface.set_index('P1').P1_IRES.apply(comma_to_list).explode().apply(dash_to_list).explode().dropna()
interface = pd.concat([p2,p1]).reset_index()
interface.columns=['uid', 'pos']
interface['gene']=interface.uid.apply(lambda x: uniprot_to_gene[x] if x in uniprot_to_gene else np.nan)
interface=interface.dropna()
interface['interface'] = True
#%%
slim['pos'] = [np.arange(r.start, r.end) for i,r in slim.iterrows()]
slim = slim.explode('pos')
slim['slim']=True
from statannotations.Annotator import Annotator
#%%
score_stats = pd.read_feather("./data.feather")
#%%
# score_stats = pd.read_csv('rank_all_cosmic/genes.txt.scores.txt')
score_stats = score_stats.merge(slim, on=['pos', 'gene'], how='left').merge(interface, on=['pos', 'gene'], how='left')[['scores', 'gene', 'pos', 'slim', 'interface', ]].fillna(False)
#%%
score_stats['group'] = 'Other'
score_stats.loc[score_stats.slim==True, 'group'] = 'SLiM'
score_stats.loc[score_stats.interface==True, 'group'] = 'Interface'
score_stats.loc[(score_stats.slim==True) & (score_stats.interface==True), 'group'] = 'SLiM&Interface'
score_stats = score_stats.drop_duplicates()
#%%
order = ["Other", "SLiM", "Interface", "SLiM&Interface"]
fig, ax = plt.subplots(figsize=(6, 3))
sns.violinplot(data=score_stats, x='group', y='scores', order=order, palette='Set2', cut=0, ax=ax, linewidth=2)
# ax.set(ylim=(0, 1))               
ax.set_xlabel("")
ax.set_ylabel("ES score distribution")

pairs=[("SLiM", "Other"), ("Interface", "Other"), ("SLiM&Interface", "Other")]
annot = Annotator(ax, pairs, data=score_stats, x='group', y='scores', order=order)
annot.configure(test='t-test_welch', pvalue_format_string="{:.2e}", text_format='full', loc='outside', show_test_name=False, line_height=0.005, text_offset=0.0005, line_offset=0.0005)
annot.apply_test()
ax, test_results = annot.annotate()

# plt.legend(loc='upper right', title='')
# plt.savefig(arg.input+"/Comparison.png",dpi=300, bbox_inches='tight')
# plt.close()
#%%
cosmic = pd.read_feather("./cosmic.feather")
# %%
cosmic['mutation'] = 'Non hotspot'
cosmic['mutation'].loc[cosmic['mut.posfrac']>0.1] = 'Hotspot'

# %%
from statannotations.Annotator import Annotator
fig, ax = plt.subplots(figsize=(3,3))   
order = ["Hotspot", "Non hotspot"]
cosmic['es'] = (cosmic['es']-cosmic['es'].min())/(cosmic['es'].max()-cosmic['es'].min())
ax = sns.violinplot(data=cosmic, x='mutation', y='scores', ax=ax, cut=0, order=order,palette='Set2', linewidth=2)

pairs=[
                        ("Hotspot", "Non hotspot"),
                        # ("Hotspot", "Not mutated"),
                        # ("Not mutated", "Non hotspot"), 
                    ]
annotator = Annotator(ax, pairs, data=cosmic, x='mutation', y='scores', order=order)
annotator.configure(test='t-test_welch', pvalue_format_string="{:.2e}", text_format='full', loc='outside', show_test_name=False)
annotator.apply_test()
ax, test_results = annotator.annotate()
ax.set_xlabel('')
# ax.set(ylim=(0, 1))
ax.set_ylabel('Mutation ES score distribution')
# plt.legend(loc='upper right', title='')
plt.savefig("final_plots/Comparison.png", dpi=300, bbox_inches='tight')
# plt.close()
# %%
