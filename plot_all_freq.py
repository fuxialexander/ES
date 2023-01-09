#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_cachedir; print(get_cachedir()) 
# import matplotlib.font_manager as font_manager; font_manager._rebuild()
plt.style.use('~/manuscript.mplstyle')
import seaborn as sns
# %%
data = pd.read_csv("ALL_hotspot.csv")
# %%
dx = data[data['Diagnosis.variant.frequency']>0].groupby(['Gene']).Sample.count()
# %%
rel = data[data['Relapse.variant.frequency']>0].groupby(['Gene']).Sample.count()
# %%
scatter = pd.concat([dx, rel], axis=1).fillna(0)
scatter.columns=['Diagnosis', 'Relapse']
scatter['diff']=scatter.iloc[:,1]-scatter.iloc[:,0]
scatter = scatter.sort_values('Relapse', ascending=True)
scatter['y'] = range(scatter.shape[0])
# %%

scatter.Diagnosis = -scatter.Diagnosis
scatter = scatter.reset_index()
# %%
scatter['Diagnosis'] = scatter.Diagnosis / 67 * 100
scatter['Relapse'] = scatter.Relapse / 67 * 100
# scatter = scatter.melt(id_vars=['Gene', 'y'], value_vars=['Diagnosis','Relapse'])
#%%
scatter = scatter.loc[scatter.Gene.isin(np.loadtxt('all_mut/genes.txt', dtype=str))]

#%%
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
fig, ax = plt.subplots(figsize = (5, 6))
# sns.scatterplot(data=scatter, x='value', y='Gene', hue='variable', ax=ax)
# plt.stem(x=scatter.value)
ax.vlines(x=0, ymin=-1, ymax= scatter.shape[0], color=GREY75, ls="-")
ax.hlines(y="Gene", xmin=0, xmax="Relapse", color=RED, ls="-", data=scatter, label=None)
ax.hlines(y="Gene", xmin="Diagnosis", xmax=0, color=BLUE, ls="-", data=scatter, label=None)
ax.hlines(y="Gene", xmin=scatter.Diagnosis.min()-1, xmax="Diagnosis", color=GREY75, ls=":", data=scatter, label=None)

ax.hlines(y="Gene", xmin=scatter.Diagnosis.min()-1, xmax="Diagnosis", color=LIGHT_RED, ls=":", data=scatter[scatter.Gene.isin(['ABL1','NT5C2'])], label=None)

ax.scatter(x="Diagnosis", y="Gene", s=100, color=LIGHT_BLUE, edgecolors=BLUE, lw=2, zorder=2, data=scatter, label="Diagnosis")
ax.scatter(x="Relapse", y="Gene", s=100, color=LIGHT_RED, edgecolors=RED, lw=2, zorder=2, data=scatter, label="Relapse")

# ax.xaxis.set_visible(False)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
ax.set_xlabel("# of samples with mutation (%)")
ax.tick_params(axis="y", bottom=True, top=True, labelbottom=True, labeltop=True, length=0)
ax.set_yticklabels(scatter.Gene.values, fontname="Arial", color='black', size=12)
ax.set_xticks([-20, 0, 20])
ax.set_xticklabels(["20%", "0%", "20%"], fontname="Arial", color='black', size=12)
plt.legend(loc='lower right', fontsize=10, bbox_to_anchor=(0.9, 0.05))
fig.savefig("figures/Fig3A.eps", bbox_inches='tight', dpi=300)
# %%
curve = data[['Gene','Predicted.protein.product','Relapse.variant.frequency', 'Diagnosis.variant.frequency']]
curve.columns = ['Gene','AA','Rel', 'Dx']
curve['AA'] = curve.AA.str.split(',')
curve = curve.explode('AA')
curve['pos'] = curve.AA.str.slice(start=1, stop=-1)
curve = curve.groupby(['Gene', 'pos', 'AA']).mean().reset_index()
curve['Rel'] = curve.Rel.astype('int')
# %%
curve[['Rel','Gene','pos']].to_csv("all_rel/mutations.txt", sep='\t', header=None, index=None)
# %%
curve[['Dx','Gene','pos']].to_csv("all_dx/mutations.txt", sep='\t', header=None, index=None)
# %%
