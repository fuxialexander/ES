#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from plot import normalize, smooth

plt.style.use('~/manuscript.mplstyle')


#%%

esm = pd.read_csv("esm_cancer_genes.csv", header=0, index_col=0)
#%%
esm['esm'] = -esm.iloc[:, 2:7].values.mean(axis=1)
esm = esm[['Mutation', 'Gene', 'esm']]
esm['pos'] = esm.Mutation.apply(lambda x: int(x[1:-1]))
#%%
enst_to_uniprot = pd.read_csv('enst_to_uniprot.tsv', sep='\t', header=0)
def load_data(data_file):
    data = pd.read_csv(data_file, index_col=0)#.dropna()
    data = pd.merge(data, data.groupby('gene').rec.sum(), on='gene', suffixes=["","sum"])
    data['recfrac'] = (data.rec/data.recsum)
    data['metric'] = 0
    data.loc[data.mutation=='Hotspot', 'metric'] = 1
    data = data.merge(enst_to_uniprot, left_on='gene', right_on='Gene_Symbol', how='left').drop('Gene_Symbol', axis=1)
    data = data.merge(esm, left_on=['gene', 'pos'], right_on=['Gene', 'pos']).drop('Gene', axis=1)
    return data
#%%
data = load_data("rank_all_cosmic/genes.txt.scores.txt")
# esm = esm[esm.Gene.isin(data.gene.unique())]
#%%

#%%
cosmic = pd.read_csv("CosmicMutantExport.missense.aachange.tsv", sep='\t', names=['gene.cosmic', 'enst_id', 'mut.rec', 'tumor', 'Mutation']).drop_duplicates()
cosmic = cosmic.groupby(['enst_id', 'Mutation'])['mut.rec'].count().reset_index()
#%%
cosmic = cosmic.merge(data, on=['enst_id', 'Mutation'])
cosmic = pd.merge(cosmic, cosmic.groupby(['gene', 'pos'])['mut.rec'].sum().reset_index(), on=['gene', 'pos'], suffixes=['', '.possum'])
cosmic = pd.merge(cosmic, cosmic.groupby(['gene'])['mut.rec'].sum().reset_index(), on=['gene'], suffixes=['', '.genesum'])
cosmic['mut.posfrac'] = cosmic['mut.rec.possum']/cosmic['mut.rec.genesum']
cosmic['mut.frac'] = cosmic['mut.rec']/cosmic['mut.rec.genesum']
#%%
scaler = StandardScaler()
pca = PCA(n_components=2)
cosmic['es'] = pca.fit_transform(scaler.fit_transform(cosmic[['esm_y', 'scores']]))[:,0]
#%%
data['es'] = pca.transform(scaler.transform(data[['esm_y', 'scores']]))[:,0]
#%%
#%%
nt5c2 = data[data.gene=='NT5C2']
nt5c2_dimer = load_data("dimer/genes.txt.scores.txt")
nt5c2['es'] = pca.transform(scaler.transform(nt5c2[['esm_y', 'scores']]))[:,0]
nt5c2_dimer['es'] = pca.transform(scaler.transform(nt5c2_dimer[['esm_y', 'scores']]))[:,0]
#%%
import pyensembl
ensembl = pyensembl.EnsemblRelease()
import pandas as pd
# %%
def query_gene_coding_sequence(gene_name, full_len, pos=None):
    gene = ensembl.genes_by_name(gene_name)[0]
    cds = [t.coding_sequence for t in gene.transcripts 
    if (t.is_protein_coding and t.protein_sequence.startswith('M') and len(t.protein_sequence) == full_len)]
    if pos is not None:
        return cds[0][pos*3:pos*3+3]
    else:
        return cds[0]
#%%
nt5c2_dimer['codon'] = [query_gene_coding_sequence('NT5C2', 561, row.pos-1) for i,row in nt5c2_dimer.iterrows()]
#%%
nt5c2_dimer['ALT'] = nt5c2_dimer.Mutation.apply(lambda x: x[-1])
#%%
nt5c2_dimer['per_aa_mutprob'] = nt5c2_dimer.apply(lambda row: pan_trans.loc[row.codon, row.ALT], axis=1)
#%%
nt5c2_dimer['es'] = normalize(nt5c2_dimer.es)

#%%
mutations = pd.read_csv('all_mut/all_mutations.txt', sep='\t')
nt5c2_rx_mut = mutations[mutations.Gene=='NT5C2']['Predicted protein product'].apply(lambda x: x.split(',')[0]).reset_index().groupby(by='Predicted protein product').count().rename({'index':'count'}, axis=1)

#%%
def t_fun(x):
    return x**2# (2*np.exp(x))/(np.exp(x) + np.exp(-x)) -1

nt5c2_to_plot = pd.merge(nt5c2_dimer, nt5c2_rx_mut, left_on='Mutation', right_index=True, how='left').fillna(0)
# nt5c2_to_plot['count'] = np.log(nt5c2_to_plot['count'].astype(int)+1)
# nt5c2_to_plot['rec'] = nt5c2_to_plot['rec']
# nt5c2_to_plot = nt5c2_to_plot[nt5c2_to_plot.es>nt5c2_to_plot.es.mean()]
nt5c2_to_plot.rename({'es': 'ES-mut', 'rec': '#Rx Mutation'}, axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(7,3))
sns.scatterplot(data=nt5c2_to_plot, x='pos', y=nt5c2_to_plot.per_aa_mutprob**2, hue='ES-mut', size='#Rx Mutation', ax=ax,sizes=(2, 100))

from matplotlib.ticker import FormatStrFormatter
ax.set_yticks(t_fun(np.array([0, 0.01, 0.02])))
ax.set_yticklabels(["{:.2f}".format(x) for x in [0, 0.01, 0.02]])
ax.set_xlim(1, 561)
# set legend to right outside of plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
ax.set_xlabel('Position')
ax.set_ylabel('Mutation probability')
ax.savefig('figures/nt5c2_mutprob.cosmic.pdf', bbox_inches='tight')
#%%
def t_fun(x):
    return x**2# (2*np.exp(x))/(np.exp(x) + np.exp(-x)) -1

def plot_curve(df, gene, output=None, df_dimer=None, score='es', with_mutprob=True):
    sns.set_theme(style="white")
    curve_data = df.loc[df.gene==gene][['scores', 'pos', 'rec',  'esm_x', 'esm_y',  'es', 'mutprob']].groupby('pos').mean().reset_index()
    curve_data[score] = t_fun(smooth(normalize(curve_data[score]),5))
    if df_dimer is not None:
        curve_data_dimer = df_dimer.loc[df_dimer.gene==gene][['scores', 'pos', 'rec', 'esm_x', 'esm_y', 'es', 'mutprob']].groupby('pos').mean().reset_index()
        curve_data[score+'_dimer'] = t_fun(smooth(normalize(curve_data_dimer[score]),5))

    
    mut_data = cosmic.loc[(cosmic.gene==gene)][['scores', 'pos', 'mut.rec', 'mut.rec.possum', 'mut.posfrac', 'mut.frac', 'mut.rec.genesum', 'esm_x', 'esm_y', 'es']]
    # mut_data['es'] = (mut_data.es - curve_data.es.min())/(curve_data.es.max() - curve_data.es.min())
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    
    if df_dimer is not None:
        sns.lineplot(data=curve_data,
                    x='pos', y=score, color='#7fcdbb', ax=ax)
        sns.lineplot(data=curve_data,
                    x='pos', y=score+'_dimer', color='#2c7fb8', ax=ax)
    else:
        sns.lineplot(data=curve_data,
                    x='pos', y=score, color='#2c7fb8', ax=ax)
    # sns.lineplot(data=curve_data,
    #                 x='AA', y='esm', color='#2c7fb8', ax=ax)
    # sns.lineplot(data=curve_data,
    #                 x='pos', y='scores', color='g', ax=ax)
    
    
    plt.hlines(y=curve_data[score].median(), xmin=1, xmax=curve_data.pos.max(), colors='black', linestyles='dotted', linewidth=1)
    ax.fill_between(curve_data.pos, 0, curve_data[score].median(), color='black', alpha=0.1)

    if with_mutprob:
        if df_dimer is not None:
            mutprob = curve_data.mutprob.copy()
            mutprob[mutprob<np.quantile(mutprob.values, 0.9)]=0
        else:
            mutprob = curve_data.mutprob.copy()
            mutprob[mutprob<np.quantile(mutprob.values, 0.9)]=0
        # sns.scatterplot(data=mut_data[mut_data['mut.posfrac']>0.1], x='pos', y='es', linewidth=0, color='#dd1c77', ax=ax, s=30, alpha=0.5)
        plt.scatter(x=curve_data.pos[mutprob>0], y=mutprob[mutprob>0]/mutprob[mutprob>0]-0.02, c='grey', alpha=1, s=10)
    
    ax2 = ax.twinx()
    
    # plt.vlines(x=mut_data.pos, ymin=0, ymax=mut_data['mut.rec.possum'], colors='#dd1c77', alpha=1)
    if df_dimer is not None:
        plt.vlines(x=curve_data.pos, ymin=0, ymax=curve_data_dimer.rec, colors='#dd1c77', alpha=1)
    else:
        plt.vlines(x=curve_data.pos, ymin=0, ymax=curve_data.rec, colors='#dd1c77', alpha=1)
    if df_dimer is not None:
        legend_elements = [
            Line2D([0], [0], color='w', lw=4, label=gene),
            Line2D([0], [0], color='#2c7fb8', lw=4, label='Dimer ES Score'),
            Line2D([0], [0], color='#7fcdbb', lw=4, label='Monomer ES Score'),
        Line2D([0], [0], color='#dd1c77', lw=4, label='Mutations')]
        plt.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.45, 1.1))
    else:
        legend_elements = [
        Line2D([0], [0], color='w', lw=4, label=gene),
        Line2D([0], [0], color='#2c7fb8', lw=4, label='ES Score'),
        Line2D([0], [0], color='#dd1c77', lw=4, label='Mutations')]
        plt.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1.1))
    
    ax.set_ylabel('ES Score')
    ax.set_xlim(1, curve_data.pos.max())
    ax.set_ylim(0, 1)
    if df_dimer is not None:
        ax2.set_ylim(0, curve_data_dimer.rec.max())
    else:
        ax2.set_ylim(0, curve_data.rec.max())
    ax.set_xlabel("")
    ax2.set_ylabel('Recurrence')
    ax2.set_xlabel("AA")
    
    from matplotlib.ticker import FormatStrFormatter
    ax.set_yticks(t_fun(np.array([0, 0.4,0.6,0.8,1.0])))
    ax.set_yticklabels(["{:.2f}".format(x) for x in [0, 0.4,0.6,0.8,1.0]])
    if output:
        fig.savefig(output+"/"+gene+'.png')
    plt.show()
    return curve_data
#%%
plot_curve(data, 'FOXL6', output='final_plots_pos', score='scores', with_mutprob=True)
#%%
plot_curve(data,  'TP53', output='final_plots_pos', score='scores')
#%%
plot_curve(data, 'MPL', output='final_plots_pos', score='scores')
#%%
plot_curve(data, 'XPO1', output='final_plots_pos', score='es')
# output='final_plots_pos'
# for g in df.head(20).gene:
#     plot_curve(data, g, output=output, score='scores')

# [['es', 'ALT', 'Amino acid position']].pivot_table(index='ALT', columns='Amino acid position', values='PM Score')
#%%
cosmic['metric'] = 0
cosmic.loc[cosmic['mut.frac']>0.1, 'metric'] = 1
#%%
import os
from tqdm import tqdm
to_uniprot = pd.read_csv("uniprot_to_genename.txt", sep='\t').set_index("To").to_dict()["From"]
def extract_esm_from_row(row):
    ref = row['Mutation'][0]
    alt = row['Mutation'][-1]
    pos = row['Mutation'][1:-1]
    header = ref + ' ' + pos
    if os.path.exists("esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+to_uniprot[row.gene]+"_LLR.csv"):
        if 'esm_file_'+row.gene in globals():
            return globals()['esm_file_'+row.gene].loc[alt, header]
        else:
            globals()['esm_file_'+row.gene] = pd.read_csv("esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+to_uniprot[row.gene]+"_LLR.csv", index_col=0)
            return globals()['esm_file_'+row.gene].loc[alt, header]

for i, row in tqdm(cosmic.iterrows()):
    try:
        cosmic.loc[i, 'esm_new'] = -extract_esm_from_row(row)
    except:
        continue

#%%
def plot_score(ctat, score, label, metric = 'metric', reverse=False):
    plt.subplots(figsize=(5,5))
    df = ctat.loc[~ctat[metric].isna()]
    if len(score) > 1:
        for s in score:
            df = df.loc[~df[s].isna()]

        for i, s in enumerate(score):
            if reverse:
                fpr, tpr, _ = roc_curve(df[metric], 1-df[s].astype(float))
            else:
                fpr, tpr, _ = roc_curve(df[metric], df[s].astype(float))
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label= label[i] + "(area = %0.3f)" % roc_auc)
            # plt.xlim((0,0.2))
            # plt.ylim((0,0.6))
            plt.legend(loc='lower right', fontsize=12)
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
        plt.plot(fpr, tpr, label=label[i] + "(area = %0.3f)" % roc_auc)
        plt.legend(loc='lower right', fontsize=12)
        # plt.xlim((0,0.2))
        # plt.ylim((0,0.6))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    plt.show()
    return roc_auc
#%%
# all_df_auroc = all_df[['scores', 'pos', 'rec', 'recfrac', 'esm', 'es', 'metric','gene']].groupby(['gene','pos']).mean().reset_index()
# all_df_auroc['metric'] = all_df_auroc['recfrac']>0.1

plot_score(cosmic, ['scores', 'es',  'esm_y', 'esm_new'], ['ES-p', 'ES-mut',  'ESM1v-mut', 'ESM1b-mut'], metric='metric')
#%%
cosmic['es'] = normalize(cosmic.es)
cosmic['weighted_score'] = normalize(cosmic.es.astype(float) * cosmic['mut.frac'])
#%%
perm_dist = []
cosmic_permuted = cosmic.copy()
for i in range(100):
    cosmic_permuted['es'] = np.random.permutation(cosmic_permuted['es'])
    cosmic_permuted['weighted_score'] = normalize(cosmic_permuted.es.astype(float) * cosmic_permuted['mut.frac'])
    df_perm = cosmic_permuted.groupby('gene').weighted_score.sum().reset_index().sort_values('weighted_score')
    df_perm = df_perm.sort_values('weighted_score', ascending=False)
    df_perm = pd.merge(df_perm, cosmic_permuted.groupby('gene')['mut.rec.genesum'].mean(), on='gene')
    df_perm['rec'] = np.log10(df_perm['mut.rec.genesum'])
    df_perm['weighted_score'] = normalize(df_perm.weighted_score.values)
    df_perm['weighted_score'] = t_func2(df_perm.weighted_score.values)
    perm_dist.append(df_perm[['weighted_score', 'rec']])

perm_dist = pd.concat(perm_dist)
#%%
# data = data.sort_values('recfrac', ascending=False).groupby('gene').head(10)
# cosmic = cosmic[cosmic.gene.isin(cosmic.groupby('gene').recfrac.max()[cosmic.groupby('gene').recfrac.max()>0.1].index)]
# cosmic = pd.read_feather("cosmic.feather")
#%%
# cosmic['weighted_score'] = cosmic.esm.astype(float) * cosmic['mut.frac']

#%%
# def get_auroc(x, y):
#     fpr, tpr, _ = roc_curve(x, y)
#     return auc(fpr, tpr)

# auroc_rank = pd.DataFrame(cosmic.groupby('gene').apply(lambda x: get_auroc(x.metric,x.es)).sort_values(ascending=False).dropna(), columns=['auroc'])
# auroc_rank = auroc_rank.merge(cosmic.groupby('gene')['metric'].sum(), left_index=True, right_index=True)

# # df = pd.merge(df, data[data.mutation!='Not mutated'].groupby('gene').gene.count(), left_on='gene', right_index=True, suffixes=["","len"])
# #%%
# # auroc_rank['rec'] = np.log10(auroc_rank['mut.rec.genesum'])

# fig, ax = plt.subplots(figsize=(5,5))
# sns.scatterplot(x="auroc", y='metric', data=auroc_rank)
# TEXTS = []
# for i in range(auroc_rank.shape[0]):
#     if i<20:
#         y = auroc_rank.iloc[i].metric
#         x = auroc_rank.iloc[i].auroc
#         text = auroc_rank.iloc[i].name
#         TEXTS.append(ax.text(x, y, text, fontsize=12))
# adjust_text(
#     TEXTS, 
#     autoalign='x',
#     # only_move={'points':'x', 'text':'x'},
#     # force_text=(2, 2),
#     # expand_points=(5, 2),
#     arrowprops=dict(
#         arrowstyle="-", 
#         lw=0.5,
#     ),
#     ax=fig.axes[0]
# )
# # ax.set_ylabel("")
# # ax.set_yticks([])
# ax.set_xlabel("Hotspot AUROC")
# %%
# df['final']=normalize(df.weighted_score)
#%%
def t_func2(x):
    return 2*x**4/(x**4+1)
#%%
df = cosmic.groupby('gene').weighted_score.sum().reset_index().sort_values('weighted_score')
df = df.sort_values('weighted_score', ascending=False)
df['Ranks']=range(df.shape[0])
df = pd.merge(df, cosmic.groupby('gene')['mut.rec.genesum'].mean(), on='gene')
df['rec'] = np.log10(df['mut.rec.genesum'])
#%%
df['weighted_score'] = normalize(df.weighted_score.values)
df['weighted_score'] = t_func2(df.weighted_score.values)
#%%
fig, ax = plt.subplots(figsize=(5,5))

sns.kdeplot(data=perm_dist.reset_index(), y='weighted_score', x='rec', fill=True, level=4, ax=ax)
sns.scatterplot(data=df, y='weighted_score', x='rec',s=10,ax=ax, linewidth=0, alpha = 0.7)
TEXTS = []
for i in range(df.shape[0]):
    if i<20 and df.iloc[i].rec>1:
        print(df.iloc[i].gene)
        y = df.iloc[i].weighted_score
        x = df.iloc[i].rec
        text = df.iloc[i].gene
        TEXTS.append(ax.text(x, y, text, fontsize=12))


adjust_text(
    TEXTS, 
    autoalign='y',
    arrowprops=dict(
        arrowstyle="-", 
        lw=0.5
    ),
    ax=fig.axes[0]
)
ax.set_ylabel("Average ES score")
ax.set_xlabel("Log10 Recurrence")
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels(["{:.2f}".format(x) for x in t_func2(np.array([0, 0.2,0.4,0.6,0.8,1.0]))])
fig.savefig("figures/rank_plot.png", dpi=300)
#%%
from scipy.stats import spearmanr
spearmanr(df.weighted_score[df.weighted_score>0.05], df.rec[df.weighted_score>0.05])
#(0.4610747235233232, 3.1406662589515234e-05)
#%%
pearsonr(df.weighted_score[df.weighted_score<0.5], df.rec[df.weighted_score<0.5])
# (-0.32297329679021064, 4.898829270172951e-06)
# %%
df = df.sort_values('weighted_score', ascending=False)
dftop = df.iloc[0:20]
GREY94 = "#f0f0f0"
GREY75 = "#bfbfbf"
GREY65 = "#a6a6a6"
GREY55 = "#8c8c8c"
GREY50 = "#7f7f7f"
GREY40 = "#666666"
LIGHT_BLUE = "#b4d1d2"
DARK_BLUE = "#242c3c"
BLUE = "#4a5a7b"
LIGHT_RED = "#ff846b"
RED = "#c43d21"
WHITE = "#FFFCFC" # technically not pure white
fig, ax = plt.subplots(figsize = (4.5, 4.5))
# sns.scatterplot(data=scatter, x='value', y='Gene', hue='variable', ax=ax)
# plt.stem(x=scatter.value)
ax.hlines(y="gene", xmin=0, xmax="weighted_score", color=RED, ls="-", data=dftop, label=None)
# ax.hlines(y="gene", xmin=df.Diagnosis.min()-1, xmax="Diagnosis", color=GREY75, ls=":", data=scatter, label=None)
ax.vlines(x=0, ymin=-1, ymax= dftop.shape[0], color=GREY40, ls="-")
ax.scatter(x="weighted_score", y="gene", s=100, color=LIGHT_RED, edgecolors=RED, lw=2, zorder=2, data=dftop, label=None)
ax.set_ylim(-1, dftop.shape[0])
# ax.xaxis.set_visible(False)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Average ES score")
ax.tick_params(axis="y", bottom=True, top=True, labelbottom=True, labeltop=True, length=0)
ax.set_yticklabels(dftop.gene.values, fontname="Arial", color='black', size=12)
# ax.set_xticks([-20, 0, 20])
ax.invert_yaxis()
# ax.set_xticklabels(["20%", "0%", "20%"], fontname="Arial", color='black', size=12)
plt.legend(loc='lower right', fontsize=12, bbox_to_anchor=(0.9, 0.05))
# %%
output='final_plots_pos.with_mutprob.es'
if not os.path.exists(output):
    os.mkdir(output)
for g in dftop.gene:
    plot_curve(data, g, output=output, score='scores')
#%%
output='final_plots_pos.without_mutprob.es'
if not os.path.exists(output):
    os.mkdir(output)
for g in dftop.gene:
    plot_curve(data, g, output=output, score='scores', with_mutprob=False)

# %%
# plot_curve(data, "NT5C2", output=output)

# %%
cosmic.to_feather('cosmic.feather')
nt5c2_dimer.to_feather('nt5c2_dimer.feather')
# %%
data.to_feather('data.feather')
# cosmic.to_feather('data.feather')
# %%
cosmic = pd.read_feather('cosmic.feather')
data = pd.read_feather('data.feather')
def get_defattr(data, score = 'es', dimer=False):
    if dimer:
        print('\n\t'.join(['attribute: es_score_dimer\nrecipient: residues']+['/A:' + str(i+1) + '\t' + data[score].astype(str)[i] for i in range(data.shape[0])] +
                          ['/B:' + str(i+1) + '\t' + data[score].astype(str)[i] for i in range(data.shape[0])]))
    else:
        print('\n\t'.join(['attribute: es_score_monomer\nrecipient: residues']+['/A:' + str(i+1) + '\t' + data[score].astype(str)[i] for i in range(data.shape[0])]))


# get_defattr(nt5c2, dimer=True)
# %%
nt5c2_dimer_attr = plot_curve(nt5c2_dimer, "NT5C2", score='scores')
get_defattr(nt5c2_dimer_attr, dimer=True)
# %%
mpl_attr = plot_curve(data, "MPL", score='scores')
get_defattr(mpl_attr, dimer=False, score='esm')
# %%


pan_trans = pd.read_csv("./cosmic_aa_transition.csv", index_col=0)

pan_trans= pan_trans/pan_trans.sum().sum()
### cancer hotspots compare with esm1b
esm_testing = pd.read_csv("esm.testing.csv").reset_index()
# %%
esm_testing_logits = pd.read_csv("esm.testing.logits.csv")
esm_testing_logits.columns = ['id', '<cls>', '<pad>', '<eos>', '<unk>', 
                    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 
                    'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 
                    'X', 'B', 'U', 'Z', 'O', '.', '-', 
                    '<null_1>', '<mask>']
esm_testing_logits = esm_testing_logits.drop(columns=['<cls>', '<pad>', '<eos>', '<unk>', '<null_1>', '<mask>', 'id', '.', '-'])
#%%
def get_esm_score_trans(i,row):
    return (esm_testing_logits.loc[:, row.ref].iloc[i] - esm_testing_logits.loc[:, row.alt].iloc[i])*pan_trans.loc[row.ref, row.alt]

def get_esm_score(i,row):
    return (esm_testing_logits.loc[:, row.ref].iloc[i] - esm_testing_logits.loc[:, row.alt].iloc[i])
# %%
for i,row in esm_testing.iterrows():
    esm_testing.loc[i, 'esm'] = get_esm_score(i,row)
    esm_testing.loc[i, 'esmtrans'] = get_esm_score_trans(i,row)
# %%
esm_testing = pd.merge(esm_testing, data, left_on=['HGNC', 'pos'], right_on=['gene', 'pos'])
# %%
esm_testing = esm_testing[['index', 'scores', 'score', 'esm_x', 'metric', 'pos', 'gene']].drop_duplicates()
# %%
esm_testing['es'] = pca.transform(scaler.transform(esm_testing[['esm_x', 'scores']]))[:,0]
# %%
plot_score(esm_testing, ['esmtrans', 'esm'], ['esmtrans','esm'], 'score')
# %%
nt5c2_esm = pd.read_csv("/home/ubuntu/data/nt5c2/esm1b/content/ALL_hum_isoforms_ESM1b_LLR/P49902_LLR.csv", index_col=0)
# %%
fig, ax = plt.subplots(figsize=(15, 5))
sns.heatmap(nt5c2_esm.sort_index() )
# %%
all_trans = pd.read_csv("all_mut/ALL_mut_transition_matrix.csv", index_col=0)
all_trans = all_trans/all_trans.sum().sum()
# %%
score = []
for nc, col in nt5c2_esm.sort_index().iteritems():
    ref = nc.split(' ')[0]
    score.append((col.values * pan_trans.loc[ref].values).sum())

# %%
nt5c2_new_score = pd.DataFrame({'pos': np.array(range(561))+1, 'es': -np.array(score)})
# %%
nt5c2_dimer_new = nt5c2_dimer.merge(nt5c2_new_score, on='pos')
# %%

# %%
plot_curve(nt5c2_dimer_new, "NT5C2", score='es')
# %%
nt5c2_dimer_new['es'] = nt5c2_dimer_new['es'] * nt5c2_dimer_new['scores']
plot_curve(nt5c2_dimer_new, "NT5C2", score='es')
# %%
