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
cosmic = pd.read_feather("../cosmic.feather")
#%%


#%%

#%%
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
eve = eve[['Mutation', 'gene', 'EVE_scores_ASM']]
eve['EVE_scores_ASM'] = eve['EVE_scores_ASM'].astype(float)
eve = eve.dropna()
#%%
data_eve = data.merge(eve, on=['gene', 'Mutation'], how='inner')


#%%

def extract_mut_from_mut(str):
    return str[2:]
ctat = pd.read_csv("./Mutation.CTAT.3D.Scores.txt", sep='\t')
ctat['protein_change_mut'] = ctat['protein_change'].apply(extract_mut_from_mut)
ctat = ctat.merge(data_eve, left_on=['gene', 'protein_change_mut'], right_on=['gene', 'Mutation'], how='inner')
#%%
to_uniprot = pd.read_csv("../uniprot_to_genename.txt", sep='\t').set_index("To").to_dict()["From"]
def extract_esm_from_row(row):
    ref = row['protein_change_mut'][0]
    alt = row['protein_change_mut'][-1]
    pos = row['protein_change_mut'][1:-1]
    header = ref + ' ' + pos
    if os.path.exists("../esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+to_uniprot[row.gene]+"_LLR.csv"):
        if 'esm_file_'+row.gene in globals():
            return globals()['esm_file_'+row.gene].loc[alt, header]
        else:
            globals()['esm_file_'+row.gene] = pd.read_csv("../esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+to_uniprot[row.gene]+"_LLR.csv", index_col=0)
            return globals()['esm_file_'+row.gene].loc[alt, header]

for i, row in ctat.iterrows():
    try:
        ctat.loc[i, 'esm_new'] = -extract_esm_from_row(row)
    except:
        continue
#%%

ctat['metric'] = ctat['OncoKB'].apply(lambda x: 1 if (x == 'Oncogenic' or x=='Likely Oncogenic') else 0) 
#%%
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_score(ctat, score, label, colorl, metric = 'metric', reverse=False):
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
            print(fpr, tpr)
            roc_auc = auc(fpr, tpr)
            df_scatter.append((s, roc_auc))
            plt.plot(fpr, tpr, label= label[i], color=colorl[i])# + "(area = %0.3f)" % roc_auc
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim((0,0.2))
            plt.ylim((0,0.5))
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
        print(fpr, tpr)
        plt.plot(fpr, tpr, label=score[0], color=colorl[i])# + "(area = %0.3f)" % roc_auc
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.axline((1, 1), slope=1)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.show()
    plt.savefig('auroc_cosmic.pdf', dpi=300)
    return df_scatter


def plot_pr(ctat, score, label, metric = 'metric', reverse=False):
    df = ctat.loc[~ctat[metric].isna()]
    df_scatter = []
    if len(score) > 1:
        for s in score:
            df = df.loc[~df[s].isna()]

        for i,s in enumerate(score):
            if reverse:
                fpr, tpr, _ = precision_recall_curve(df[metric], 1-df[s].astype(float))
            else:
                fpr, tpr, _ = precision_recall_curve(df[metric], df[s].astype(float))
            # roc_auc = auc(fpr, tpr)
            # df_scatter.append((s, roc_auc))
            plt.plot(fpr, tpr, label= label[i])# + "(area = %0.3f)" % roc_auc
            plt.plot([0, 1], [0, 1], 'k--', color='gray')
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.legend(loc='upper left', fontsize=12)
            plt.xlabel('Recall')
            plt.ylabel('Precision')

    
    else:
        df = ctat.iloc[np.where(~df[score].isna())[0]]
        df = ctat.iloc[np.where(~df[metric].isna())[0]]
        if reverse:
            fpr, tpr, _ = roc_curve(df[metric], 1-df[score].astype(float))
        else:
            fpr, tpr, _ = roc_curve(df[metric], df[score].astype(float))
        # roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=score[0])# + "(area = %0.3f)" % roc_auc
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.axline((1, 1), slope=1)
        plt.legend(loc='upper left', fontsize=12)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.show()
    return df_scatter
# #%%
# p = plot_score(ctat, ['esm1v_t33_650M_UR90S_5'], reverse = False)
# # %%
# import numpy as np
# score = pd.read_csv("../rank_all_cosmic/genes.txt.scores.txt", sep=',')
# score['weighted_score'] = score.scores#np.log10(score.rec*score.scores+1)#/data.recsum
# # %%
# # %%
# gene_score_list = list(score.groupby('gene')['pos','weighted_score'])
# gene_score_dict = {i[0]: i[1].sort_values('pos').set_index('pos').weighted_score.to_dict() for i in gene_score_list}
# # %%

# def extract_pos_from_mut(str):
#     return int(str[3:-1])

# def extract_mut_from_mut(str):
#     return str[2:]

# def get_score_from_mut(gene, pos):
#     try:
#         s = gene_score_dict[gene][pos]
#     except:
#         s = -100
#     return s

# from tqdm import tqdm
# p_scores = []
# for i, row in tqdm(ctat.iterrows()):
#     # print(get_score_from_mut(row['gene'], extract_pos_from_mut(row['protein_change'])))
#     p_scores.append(get_score_from_mut(row['gene'], extract_pos_from_mut(row['protein_change'])))

# #%% 
# import numpy as np
# ctat['p_score'] = np.array(p_scores)

# ctat = ctat[ctat.p_score>-10]

# #%%
# plot_score(ctat[ctat.gene!="KRAS"], ['p_score'], reverse = False)
# auroc = pd.DataFrame.from_dict({i:get_auroc(ctat[ctat.gene==i], ['p_score'], reverse = False) for i in ctat.gene.unique()}, orient='index')
# # %%
# ctat['score (transFIC)'] = ctat['score (transFIC)'].fillna(ctat['score (transFIC)'].mean())
# ctat['score (CHASM)'] = ctat['score (CHASM)'].fillna(ctat['score (CHASM)'].mean())
# ctat['score (fathmm)'] = -ctat['score (fathmm)'].fillna(ctat['score (fathmm)'].mean())
# ctat['score (CanDrA)'] = ctat['score (CanDrA)'].fillna(ctat['score (CanDrA)'].mean())
# ctat['p_score'] = ctat['p_score'].fillna(ctat['p_score'].mean())
# #%%
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# input_data = scaler.fit_transform(ctat[[
#     'score (transFIC)', 'score (CHASM)', 'score (fathmm)', 'score (CanDrA)', 
#     'p_score',
#     'esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3', 'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5']])
# old_input_data = scaler.fit_transform(ctat[['score (transFIC)', 'score (CHASM)', 'score (fathmm)', 'score (CanDrA)',]])
# pca = PCA(n_components=2)
# new_score = pca.fit_transform(input_data)[:,0]
# old_score = pca.fit_transform(old_input_data)[:,0]
# ctat['CTAT-Cancer w/ p_score'] = new_score
# ctat['CTAT-Cancer'] = old_score
# #%%
# ctat['score (VEST)'] = ctat['score (VEST)'].fillna(ctat['score (VEST)'].mean())
# ctat['score (MutationAssessor)'] = ctat['score (MutationAssessor)'].fillna(ctat['score (MutationAssessor)'].mean())
# ctat['score (PolyPhen2)'] = ctat['score (PolyPhen2)'].fillna(ctat['score (PolyPhen2)'].mean())
ctat['score (SIFT)'] = -ctat['score (SIFT)'].fillna(ctat['score (SIFT)'].mean())
# ctat['score (MutationTaster)'] = -ctat['score (MutationTaster)'].fillna(ctat['score (MutationTaster)'].mean())

# #%%
# ctat['esm1v_t33_650M_UR90S_1'] = -ctat['esm1v_t33_650M_UR90S_1'].astype(float)
# ctat['esm1v_t33_650M_UR90S_2'] = -ctat['esm1v_t33_650M_UR90S_2'].astype(float)
# ctat['esm1v_t33_650M_UR90S_3'] = -ctat['esm1v_t33_650M_UR90S_3'].astype(float)
# ctat['esm1v_t33_650M_UR90S_4'] = -ctat['esm1v_t33_650M_UR90S_4'].astype(float)
# ctat['esm1v_t33_650M_UR90S_5'] = -ctat['esm1v_t33_650M_UR90S_5'].astype(float)
# #%%
# ctat['esm score'] = -(ctat['esm1v_t33_650M_UR90S_1'] + ctat['esm1v_t33_650M_UR90S_2'] + ctat['esm1v_t33_650M_UR90S_3'] + ctat['esm1v_t33_650M_UR90S_4'] + ctat['esm1v_t33_650M_UR90S_5'])/5
# #%%
# input_data = scaler.fit_transform(ctat[['score (VEST)', 'score (MutationAssessor)','score (PolyPhen2)', 'score (SIFT)', 'score (MutationTaster)', 'p_score', 'esm score']])
# input_data_s_only = scaler.fit_transform(ctat[['score (VEST)', 'score (MutationAssessor)','score (PolyPhen2)', 'score (SIFT)', 'score (MutationTaster)', 'p_score']])
# # input_data_esm_only = scaler.fit_transform(ctat[['esm1v_t33_650M_UR90S_1', 'esm1v_t33_650M_UR90S_2', 'esm1v_t33_650M_UR90S_3', 'esm1v_t33_650M_UR90S_4', 'esm1v_t33_650M_UR90S_5']])
# input_data_s_esm = scaler.fit_transform(ctat[['p_score', 'esm score']])
# old_input_data = scaler.fit_transform(ctat[['score (VEST)', 'score (MutationAssessor)','score (PolyPhen2)', 'score (SIFT)', 'score (MutationTaster)']])
# pca = PCA(n_components=2)
# new_score = pca.fit_transform(input_data)[:,0]
# new_score_s_esm = pca.fit_transform(input_data_s_esm)[:,0]
# new_score_s_only = pca.fit_transform(input_data_s_only)[:,0]
# # new_score_esm_only = pca.fit_transform(input_data_esm_only)[:,0]
# old_score = pca.fit_transform(old_input_data)[:,0]
# ctat['CTAT-Population + ES Score'] = -new_score
# ctat['CTAT-Population + P Score'] = -new_score_s_only
# ctat['ES score (amino acid specific)'] = new_score_s_esm
# ctat['CTAT-Population'] = -old_score
# #%%
# ctat['ES score'] = ctat['p_score']
# fig, ax = plt.subplots(figsize=(5,5))
# plot_score(ctat, [
#     'ES score (amino acid specific)',
#     'ES score',
#     'CTAT-Population',
#     'score (VEST)',
#     'score (MutationAssessor)',
#     'score (PolyPhen2)',
#     'score (SIFT)',
# ], metric='metric', reverse = False)
# plt.show()
# plt.savefig("auroc_cosmic.png")
#%%
fig, ax = plt.subplots(figsize=(4,4))
df_scatter = plot_score(ctat, [
    'es',
    'esm_new',
    'EVE_scores_ASM',
    'score (MutationAssessor)',
    'eigen score (functional)',
    'score (VEST)',
    
    # 'score (MutationTaster)',
    # 'score (PolyPhen2)',
    # 'score (SIFT)',
], [
    'ES-mut',
    'ESM1b',
    'EVE',
    'MutationAssessor',
    'CTAT-Population',
    'VEST',
    
    # 'MutationTaster',
    # 'PolyPhen2',
    # 'SIFT',
], ['#2c7fb8', '#91bfdb', '#ddd1e6', '#d0d1e6', '#fee090', '#fee020'], metric='metric', reverse = False)
#%%
sns.scatterplot(x='esm score', y=np.log(ctat['recurrence']), data=ctat)

# from scipy.stats import spearmanr
# spearmanr(ctat['esm score'], ctat['recurrence'])
# %%
# esm_nt5c2 = esm[esm.Gene=='NT5C2']
esm_nt5c2 = pd.read_csv("./NT5C2_esm_masked_marginal.csv")
#%%
esm_nt5c2['esm'] = esm_nt5c2.iloc[:,3:].astype(float).mean(axis=1)
def extract_wt_from_mut(str):
    return str[0:1]

def extract_alt_from_mut(str):
    return str[-1:]

def extract_pos_from_mut(str):
    return int(str[1:-1])
esm_nt5c2['ALT'] = esm_nt5c2.Mutation.apply(extract_alt_from_mut)
esm_nt5c2['Amino acid position'] = esm_nt5c2.Mutation.apply(extract_pos_from_mut)
#%%
nscore = pd.read_csv("../dimer_cosmic/genes.txt.scores.txt", sep=',')
nscore['weighted_score'] = nscore.scores
esm_nt5c2 = esm_nt5c2.merge(nscore, left_on=['Gene', 'Amino acid position'], right_on=['gene', 'pos']).drop_duplicates()
#%%
input_data_s_esm = scaler.fit_transform(ctat[['p_score', 'esm score']])
pca = PCA(n_components=2)
pca = pca.fit(input_data_s_esm)
#%%
esm_nt5c2.loc[:, ['scores', 'esm']] = scaler.fit_transform(esm_nt5c2[['scores', 'esm']])
esm_nt5c2['PM Score'] = pca.transform(esm_nt5c2[['scores', 'esm']])[:,0]
# new_score_esm_only = pca.fit_transform(input_data_esm_only)[:,0]
# %%
esm_nt5c2 = esm_nt5c2[['PM Score', 'ALT', 'Amino acid position']].pivot_table(index='ALT', columns='Amino acid position', values='PM Score')
# %%
import matplotlib
import seaborn as sns
sns.set_theme(style="whitegrid")
g = sns.clustermap(esm_nt5c2.fillna(0), row_cluster=False, col_cluster=False, cmap='mako_r', vmin=-20, vmax=-15, figsize=(15,6))
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.yaxis.tick_left()
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
g.ax_heatmap.tick_params(labelsize=10, left=False)
# %%
import matplotlib as mpl
# mpl.rcParams['font.size'] = 20
# sns.set(font="Arial")
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(15,5))
ax = sns.heatmap(esm_nt5c2.fillna(0), cmap='mako', cbar_kws={'label': 'PM score','pad':0.005}, ax=ax)
ax.yaxis.tick_left()
ax.set_ylabel("")
ax.tick_params(labelsize=16, left=False)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,va="center")
ax.set_xticks(np.arange(50, len(esm_nt5c2.columns)+1, 50))
ax.set_xticklabels(np.arange(50, len(esm_nt5c2.columns)+1, 50), rotation=0)
plt.xlabel('Amino acid position', fontsize=18)
ax.figure.axes[-1].set_ylabel("PM Score",fontsize=18)
plt.savefig("nt5c2_pm_score.png",dpi=300, bbox_inches='tight')
# %%
import matplotlib.pyplot as plt
from matplotlib import get_cachedir; print(get_cachedir()) 
# import matplotlib.font_manager as font_manager; font_manager._rebuild()
plt.style.use('~/manuscript.mplstyle')

# %%
sns.lineplot(x=np.arange(esm_nt5c2.columns.shape[0])+1, y=esm_nt5c2.mean(0))



# %%
