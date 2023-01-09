#%% 
from cProfile import label
from turtle import width
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
from matplotlib import get_cachedir; print(get_cachedir()) 
# import matplotlib.font_manager as font_manager; font_manager._rebuild()
plt.style.use('~/manuscript.mplstyle')
#%%
# esm = pd.read_csv("../esm_cancer_genes.csv")
data = pd.read_feather("../data.feather")

from tqdm import tqdm
genename_to_uniprot_ac = pd.read_csv("../uniprot_ac_to_genename.txt", sep='\t').set_index("To").to_dict()["From"]
def get_eve_data(gene):
    if gene in genename_to_uniprot_ac:
        df = pd.read_csv('../eve/variant_files/'+genename_to_uniprot_ac[gene]+'.csv', dtype=str)
        df['Mutation'] = df['wt_aa'] + df['position'] + df['mt_aa']
        df['gene'] = gene
        return df
eve = pd.concat([get_eve_data(gene) for gene in tqdm(data['gene'].unique())])
#%%
# eve = eve[['Mutation', 'gene', 'EVE_scores_ASM']]
# eve['EVE_scores_ASM'] = eve['EVE_scores_ASM'].astype(float)
# eve = eve.dropna()
#%%
data_eve = data.merge(eve, on=['gene', 'Mutation'], how='inner')

# %%
data_eve = data_eve.dropna(subset=['ClinVar_ClinicalSignificance'])
# %%
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_score(ctat, score, label, metric = 'metric', reverse=False):
    df = ctat.loc[~ctat[metric].isna()]
    df_scatter = []
    if len(score) > 1:
        for s in score:
            df = df.loc[~df[s].isna()]

        for i,s in enumerate(score):
            if reverse:
                fpr, tpr, _ = roc_curve(df[metric], 1-df[s].astype(float))
            else:
                fpr, tpr, _ = roc_curve(df[metric], df[s].astype(float))
            roc_auc = auc(fpr, tpr)
            df_scatter.append((s, roc_auc))
            plt.plot(fpr, tpr, label= label[i])# + "(area = %0.3f)" % roc_auc
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.legend(loc='upper left', fontsize=12)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

    
    else:
        df = ctat.iloc[np.where(~df[score].isna())[0]]
        df = ctat.iloc[np.where(~df[metric].isna())[0]]
        if reverse:
            fpr, tpr, _ = roc_curve(df[metric], 1-df[score].astype(float))
        else:
            fpr, tpr, _ = roc_curve(df[metric], df[score].astype(float))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=score[0])# + "(area = %0.3f)" % roc_auc
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.axline((1, 1), slope=1)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.show()
    return df_scatter

# %%

data_eve['metric'] = data_eve['ClinVar_ClinicalSignificance'].apply(lambda x: 1 if (x in ['Pathogenic',
       'Likely pathogenic', 'Pathogenic/Likely pathogenic', 'Likely pathogenic, other',
       'Pathogenic/Likely pathogenic, other', 'Pathogenic/Likely pathogenic, drug response',
       'Likely pathogenic, drug response', 'Likely pathogenic, risk factor',
       'Pathogenic/Likely pathogenic, risk factor',]) else 0) 

# %%
plot_score(data, ['es', 'esm'], ['ES', 'ESM'])
# %%
