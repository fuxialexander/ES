#%%
import pandas as pd 

aachange = pd.read_csv("CosmicMutantExport.missense.aachange.tsv", sep="\t", header=None)#.iloc[:,4]
aachange.loc[:,1]=aachange.loc[:,1].str.split(".").str[0]


#%%
enst2ccds = pd.read_csv("CCDS2Sequence.current.txt", sep="\t")
enst2ccds = enst2ccds[enst2ccds["source"] == "EBI"]
enst2ccds['nucleotide_ID'] = enst2ccds['nucleotide_ID'].str.split(".").str[0]
enst2ccds = enst2ccds[['nucleotide_ID','#ccds']].drop_duplicates().set_index('nucleotide_ID').to_dict()['#ccds']
# %%
aachange['CCDS'] = aachange.loc[:,1].apply(lambda x: enst2ccds[x] if x in enst2ccds else None)
#%%
aachange['ref'] = aachange.loc[:,4].apply(lambda x: x[0])
aachange['alt'] = aachange.loc[:,4].apply(lambda x: x[-1])
aachange['pos'] = aachange.loc[:,4].apply(lambda x: x[1:-1])
aachange = aachange[aachange.ref.isin(list("ACDEFGHIKLMNPQRSTVWY")) & aachange.alt.isin(list("ACDEFGHIKLMNPQRSTVWY"))]
aachange = aachange[~aachange['pos'].str.contains('[A-Za-z]')]

#%%
aachange['pos'] = aachange.pos.apply(lambda x: int(x))

#%%
# aachange = pd.DataFrame.from_records(aachange.apply(lambda x: (x[0], x[-1])).values, columns=['ref','alt'])
# %%
from Bio import SeqIO
import gzip 

ccds_dict = {}
with gzip.open('CCDS_nucleotide.current.fna.gz', 'rt') as f:
    for record in SeqIO.parse(f, "fasta"):
        ccds_dict[record.id.split('|')[0]] = record.seq

#%%
def get_mut_seq(ccds_id, pos):
    if ccds_id in ccds_dict:
        return str(ccds_dict[ccds_id][(pos-1)*3:pos*3])
    else:
        return None
#%%
aachange['mut_nt'] = aachange[['CCDS', 'pos']].apply(lambda x: get_mut_seq(x.CCDS,x.pos), axis=1)
#%%
aachange = aachange[aachange['mut_nt'].notnull()]

#%%
aachange = aachange.groupby(['mut_nt','alt']).size().sort_values(ascending=False).reset_index().pivot(index='mut_nt', columns='alt').fillna(0)#.to_csv("cosmic_aa_transition.csv")
# %%
import seaborn as sns
sns.heatmap(aachange, cmap="YlGnBu")
# %%
aachange.to_csv("cosmic_aa_transition.csv")
# %%
