#%%
import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import seaborn as sns  
import matplotlib.pyplot as plt
from dash.dependencies import Output, Input
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

def normalize(x):
    new_x = x-x.min()
    return new_x/new_x.max()

def smooth(arr, kernel_size=10, method='gaussian'):
    if method == 'conv':
        kernel = np.ones(kernel_size) / kernel_size
        f = np.convolve(arr, kernel, mode='same')
    elif method == 'gaussian':
        f = gaussian_filter1d(arr, sigma=kernel_size/2)
    return f

# load the dataframe
data = pd.read_feather("../data.feather")
cosmic = pd.read_feather("../cosmic.feather")
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
        plt.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.45, 1.2), frameon=False)
    else:
        legend_elements = [
        Line2D([0], [0], color='w', lw=4, label=gene),
        Line2D([0], [0], color='#2c7fb8', lw=4, label='ES Score'),
        Line2D([0], [0], color='#dd1c77', lw=4, label='Mutations')]
        plt.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1.2), frameon=False)
    
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
    fig.savefig('plot.png', bbox_inches='tight', dpi=300)
    return 
#%%

#%%
# create the app
app = dash.Dash(__name__)
server = app.server

import base64

app.layout = html.Div(
    className='container mt-3',
    children=[# include Bootstrap CSS and JavaScript files
    html.Link(
        rel="stylesheet",
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
        integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk",
        crossOrigin="anonymous"
    ),
    html.Script(
        src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
        integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI",
        crossOrigin="anonymous"
    ),

    html.Div(className='container', children=[
        # add dropdown menu to select gene
        dcc.Dropdown(
            id='gene-select',
            options=[
                {'label': gene, 'value': gene}
                for gene in data['gene'].unique()
            ],
            value=data['gene'].unique()[0]
        ),
        # add placeholder for plot
        html.Img(id='plot-img', style={'width': '100%'})
    ])
])
@app.callback(
    Output('plot-img', 'src'),
    [Input('gene-select', 'value')]
)
def update_plot(gene):
    # generate plot using selected gene
    plot_curve(data, gene)
    # encode plot image as base64 string
    with open('plot.png', 'rb') as f:
        plot_base64 = base64.b64encode(f.read()).decode()
    # return static image source
    return 'data:image/png;base64,{}'.format(plot_base64)

if __name__ == '__main__':
    app.run_server(debug=True, port=4568)
# %%
