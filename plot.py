# %%
import argparse
from email.policy import default
import gzip
from locale import normalize
import os
import sys
from os.path import exists
import matplotlib.pyplot as plt
plt.style.use('~/manuscript.mplstyle')
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser
from Bio import SeqIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.ndimage import gaussian_filter1d
from statannot import add_stat_annotation
from tqdm import tqdm
bioparser = PDBParser()
#%%
import pyensembl
ensembl = pyensembl.EnsemblRelease()
# %%
def query_gene_coding_sequence(gene_name, full_len, pos=None):
    gene = ensembl.genes_by_name(gene_name)[0]
    cds = [t.coding_sequence for t in gene.transcripts 
    if (t.is_protein_coding and t.protein_sequence.startswith('M') and len(t.protein_sequence) == full_len)]
    if len(cds) == 0:
        return None
    if pos is not None:
        return cds[0][pos*3:pos*3+3]
    else:
        return cds[0]
# %%
query_gene_coding_sequence('NBPF20', 5207)
#%%

def extract_wt_from_mut(str):
    return str[0:1]

def extract_alt_from_mut(str):
    return str[-1:]

def extract_pos_from_mut(str):
    return int(str[1:-1])

class Gene:
    """A gene to plot"""

    def __init__(
            self, 
            gname, 
            lddt, 
            to_uniprot, 
            mut, 
            sec_str_gap=3, 
            interaction_threshold=15, 
            smooth_kernel=10,
            smooth_method='conv', 
            hotspot_threshold=0.1,
            oncogene=False,
            tsg=False,
            threed=False,
            dimer=False,
            disease_transition=False):
        self.name = gname
        
        # if gname !='BCR_ABL1' or gname not in to_uniprot:
        #     print("gname error")
        #     raise ValueError
        self.oncogene=oncogene
        self.tsg=tsg
        if gname=='BCR_ABL1':
            self.uid='BCR_ABL1'
            self.lddt = np.load("dimer_structures/BCR_ABL1.lddt.npy")/100
        else:
            self.uid = to_uniprot[gname]
            self.lddt = lddt[self.uid]/100
        self.esm = np.zeros_like(self.lddt)
        if disease_transition:
            disease_transition_table = pd.read_csv(disease_transition, index_col=0)
            # print(disease_transition_table)
            self.wt_marginal_dict = (disease_transition_table.sum(1)/disease_transition_table.sum(1).sum()).to_dict()
            # print(self.wt_marginal_dict)
        
        if self.name == 'BCR_ABL1':
            self.wt_nt = np.array([str(SeqIO.read('BCR_ABL1.ccds.fa', 'fasta').seq)[i*3:(i+1)*3] for i in range(0, 1530)])
            self.mut_prob = np.array([self.wt_marginal_dict[x] if x in self.wt_marginal_dict else 0 for x in self.wt_nt])

        # elif os.path.exists("esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+self.uid+"_LLR.csv"):
        #     esm = pd.read_csv("esm1b/content/ALL_hum_isoforms_ESM1b_LLR/"+self.uid+"_LLR.csv", index_col=0)
        #     self.wt_nt = np.array([query_gene_coding_sequence(gname, esm.columns.shape[0], i) for i in range(esm.columns.shape[0])])
        #     print(self.wt_nt)
        #     if None in self.wt_nt:
        #         print("wt_nt error")
        #         raise ValueError
        #     esm_score = []
        #     for nc, col in esm.sort_index().iteritems():
        #         pos = int(nc.split(' ')[1])
        #         esm_score.append((-col.values * (disease_transition_table/disease_transition_table.sum().sum()).loc[self.wt_nt[pos-1]].values).sum())
        #     self.esm_new = np.array(esm_score)
        #     self.wt_pos = pd.Series(esm.columns).apply(lambda x: x.split(' ')[1])
        #     self.mut_prob = np.array([self.wt_marginal_dict[x] for x in self.wt_nt])

        if os.path.exists("esm_ALL_hotspot/"+gname+".csv"):
            esm = pd.read_csv("esm_ALL_hotspot/"+gname+".csv")
            full_len = int(pd.read_csv("esm_ALL_hotspot/"+gname+".csv").iloc[-1,1][1:-1])
            print(full_len)
            self.wt_nt = np.array([query_gene_coding_sequence(gname, full_len, i) for i in range(full_len)])
            if None in self.wt_nt:
                print("wt_nt error:" + gname)
                self.mut_prob = np.zeros_like(self.wt_nt)
                # raise ValueError
            # self.wt_pos = pd.Series(esm.columns).apply(lambda x: x.split(' ')[1])
            else:
                self.mut_prob = np.array([self.wt_marginal_dict[x] for x in self.wt_nt])

            esm['esm'] = esm.iloc[:,3:].astype(float).mean(axis=1)
            esm['ALT'] = esm.Mutation.apply(extract_alt_from_mut)
            esm['REF'] = esm.Mutation.apply(extract_wt_from_mut)
            esm['Amino acid position'] = esm.Mutation.apply(extract_pos_from_mut)
            esm = esm[['esm', 'ALT', 'Amino acid position']].pivot_table(index='ALT', columns='Amino acid position', values='esm').fillna(0)
            self.esm = normalize(-esm.mean(0).values)
            # self.esmstd = normalize(esm.std(0).values)   
            # esm = esm[['esm', 'ALT', 'Amino acid position']].pivot_table(index='ALT', columns='Amino acid position', values='esm').fillna(0)
            # self.esm = normalize(-esm.mean(0).values)
        if self.lddt is None:
            print("lddt error")
            raise ValueError
        self.length = len(self.lddt)
        if self.lddt.size > 2700:
            print("lddt size error: " + gname)
            raise ValueError
        
        self.smooth_kernel = smooth_kernel
        self.smooth_method = smooth_method
        self.hotspot_threshold = hotspot_threshold
        self.smooth_lddt = smooth(self.lddt, smooth_kernel, smooth_method)
        df = mut.loc[mut.gene == gname].sort_values(
            'pos').reset_index(drop=True)
        if df.shape[0]!=0:
            self.mut = df.loc[df.pos-1 < self.length]
            self.mut_idx = self.mut.pos.values
            self.mut_rec = self.mut.recurrence.values
        self.sec_str_gap = sec_str_gap

        # self.cif_df = self.get_cif_df()
        self.grad = square_grad(self.smooth_lddt)
        self.grad = np.clip(self.grad, np.quantile(self.grad, 0.2), np.quantile(self.grad, 0.8))
        # self.ss_lddt = self.secondary_structure_mean_lddt()
        self.pairwise_distance = self.get_pairwise_distance(dimer=dimer)
        if threed:
            self.final_score = self.get_final_score_gated_grad_3d(smooth_kernel, smooth_method, interaction_threshold)
        else: 
            self.final_score = self.get_final_score_gated_grad(smooth_kernel, smooth_method)
        
        if df.shape[0]!=0:
            h_threshold = max(1, self.mut_rec.sum()*hotspot_threshold)
            h_idx = np.where(self.mut_rec > h_threshold)[0]
            nh_idx = np.where(self.mut_rec <= h_threshold)[0]
            if len(h_idx) > 0:
                self.hotspot_idx = self.mut_idx[np.where(
                    self.mut_rec > h_threshold)[0]]-1
                self.hotspot_rec = self.mut_rec[np.where(
                    self.mut_rec > h_threshold)[0]]
                self.hotspot_scores = self.final_score[self.hotspot_idx]
                self.hotspot_mut_prob = self.mut_prob[self.hotspot_idx]
                self.hotspot_esm = self.esm[self.hotspot_idx]
                self.hotspot_pos = self.hotspot_idx+1
            if len(nh_idx) > 0:
                self.non_hotspot_idx = self.mut_idx[np.where(
                    self.mut_rec <= h_threshold)[0]]-1
                self.non_hotspot_rec = self.mut_rec[np.where(
                    self.mut_rec <= h_threshold)[0]]
                self.non_hotspot_pos = self.non_hotspot_idx+1
                self.non_hotspot_mut_prob = self.mut_prob[self.non_hotspot_idx]
                self.non_hotspot_esm = self.esm[self.non_hotspot_idx]
                self.non_hotspot_scores = self.final_score[self.non_hotspot_idx]
            
            self.not_mut_idx = np.where(1-np.isin(np.arange(self.length),
                                        self.mut_idx-1) > 0)[0]
            self.not_mut_scores = self.final_score[self.not_mut_idx]
            self.not_mut_mut_prob = self.mut_prob[self.not_mut_idx]
            self.not_mut_esm = self.esm[self.not_mut_idx]
            self.not_mut_pos = self.not_mut_idx+1
        # if len(self.hotspot_idx)==0:
        #     return

    def get_cif_df(self):
        with gzip.open('structures/AF-' + self.uid + '-F1-model_v2.cif.gz', 'rt') as f:
            cif = MMCIF2Dict(filename=f)
        key_list = [
            '_struct_conf.beg_auth_seq_id',
            '_struct_conf.end_auth_seq_id',
            '_struct_conf.conf_type_id',
        ]
        df = pd.DataFrame({key: cif[key] for key in key_list})
        df.columns = ['beg', 'end', 'structure']
        df = df.loc[df.structure.isin(
            ['STRN', 'HELX_RH_AL_P', 'HELX_LH_PP_P', 'HELX_RH_3T_P', 'HELX_RH_PI_P'])]
        df['beg'] = df.beg.astype('int')
        df['end'] = df.end.astype('int')
        df = df.loc[df.end-df.beg > 1]
        df_simp = []
        last_beg, last_end = -1, -1
        last_structure = ''
        for _, row in df.iterrows():
            if last_beg > 0 and row.beg-last_end <= self.sec_str_gap and last_structure == row.structure[0:4]:
                last_end = row.end
            elif last_beg < 0:
                last_beg = row.beg
                last_end = row.end
                last_structure = row.structure[0:4]
            else:
                df_simp.append((last_beg, last_end, last_structure))
                last_beg = row.beg
                last_end = row.end
                last_structure = row.structure[0:4]

        df_simp.append((last_beg, last_end, last_structure))
        df_simp = pd.DataFrame(df_simp)
        return df_simp

    def get_interval_lddt_mean(self, beg, end):
        return self.lddt[beg-1:end-1].mean()

    def secondary_structure_mean_lddt(self):
        ss_lddt = np.zeros_like(self.lddt)
        for _, (beg, end, _) in self.cif_df.iterrows():
            ss_lddt[beg-1:end-1] = self.get_interval_lddt_mean(beg, end)
        return ss_lddt

    def get_pairwise_distance(self,dimer=False):
        if self.name == 'BCR_ABL1':
            structure = bioparser.get_structure('monomer', 'dimer_structures/' + self.name + '.pdb')
    
            model = structure[0]
            chain = model['A']
            residues = [r for r in model.get_residues()]
            whole_len = len(residues)
            chain_len = len(chain)
            distance = np.zeros((whole_len, whole_len))
            for i, residue1 in enumerate(residues):
                for j, residue2 in enumerate(residues):
                    # compute distance between CA atoms
                    try:
                        d = residue1['CA'] - residue2['CA']
                        distance[i][j] = d
                        distance[j][i] = d
                    except KeyError:
                        continue
            np.save("pairwise_interaction/" + self.name + ".npy", distance)
        elif dimer and (self.name=='NT5C2'):
            structure = bioparser.get_structure('dimer', 'dimer_structures/' + self.name + '.pdb')
            model = structure[0]
            chain = model['A']
            residues = [r for r in model.get_residues()]
            whole_len = len(residues)
            chain_len = len(chain)
            distance = np.zeros((whole_len, whole_len))
            for i, residue1 in enumerate(residues):
                for j, residue2 in enumerate(residues):
                    # compute distance between CA atoms
                    try:
                        d = residue1['CA'] - residue2['CA']
                        distance[i][j] = d
                        distance[j][i] = d
                    except KeyError:
                        continue
            distance = np.fmin(
                    distance[0:chain_len, 0:chain_len], distance[0:chain_len, chain_len:whole_len])

        else:
            if exists("pairwise_interaction/"+self.name+".npy"):
                distance = np.load("pairwise_interaction/"+self.name+".npy")
            else:
                # read structure from file

                with gzip.open('structures/AF-' + self.uid + '-F1-model_v2.pdb.gz', 'rt') as f:
                    structure = bioparser.get_structure('monomer', f)


                model = structure[0]
                chain = model['A']
                residues = [r for r in model.get_residues()]
                whole_len = len(residues)
                chain_len = len(chain)
                distance = np.zeros((whole_len, whole_len))
                for i, residue1 in enumerate(residues):
                    for j, residue2 in enumerate(residues):
                        # compute distance between CA atoms
                        try:
                            d = residue1['CA'] - residue2['CA']
                            distance[i][j] = d
                            distance[j][i] = d
                        except KeyError:
                            continue

            np.save("pairwise_interaction/" + self.name + ".npy", distance)
        return distance

    def get_final_score_gated_grad(self, smooth_kernel, smooth_method):
        f = self.grad * self.esm
        f = normalize(f)
        # f[self.lddt<50] = 0
        return f

    def get_final_score_gated_grad_3d_esm_new(self, smooth_kernel, smooth_method, interaction_threshold=20):
        f = self.grad 
        f = f * self.esm
        f[self.lddt<0.5]=0
        pairwise_interaction = self.pairwise_distance < interaction_threshold
        f = get_3d_avg(f, pairwise_interaction)
        f = normalize(f)
        return f

    def get_final_score_gated_grad_3d(self, smooth_kernel, smooth_method, interaction_threshold=20):
        f = self.grad * self.esm
        pairwise_interaction = self.pairwise_distance < interaction_threshold
        f[(self.smooth_lddt<0.5)]=0
        f = get_3d_avg(f, pairwise_interaction) 
        f = normalize(f) 
        
        return f

    def get_final_score_composite(self, interaction_threshold=10, smooth_kernel=10, smooth_method='gaussian'):
        pairwise_interaction = self.pairwise_distance < interaction_threshold
        lddt_3d = demax(get_3d_avg(self.lddt, pairwise_interaction))
        reg_score = self.ss_lddt
        reg_score[reg_score>0] = 1 - reg_score[reg_score>0]
        reg_score = demax(get_3d_max(reg_score, pairwise_interaction))
        final_score = smooth(self.grad) #* reg_score#lddt_3d #* reg_score * self.smooth_lddt
        final_score[self.lddt<0.7]=0
        final_score =  get_3d_avg(final_score, pairwise_interaction)
        final_score = normalize(smooth(final_score, 10))
        
        
        return final_score

    def set_final_score(self, score_function=get_final_score_composite, interaction_threshold=15, smooth_kernel=10, smooth_method='conv'):
        self.final_score = score_function(
            self, interaction_threshold, smooth_kernel, smooth_method)
        return self

    def plot_curve(self, output=None):
        sns.set_theme(style="white")
        curve_data = pd.DataFrame({
            "AA": np.arange(len(self.lddt)),
            # "smooth_lddt": self.smooth_lddt,
            # "esm": normalize(smooth(self.esm, 10)),
            # "grad": normalize(smooth(self.grad, 10)),
            # "esmstd": normalize(smooth(self.esmstd, 10)),
            # "ss_lddt": self.ss_lddt,
            "final_score": normalize(smooth(self.final_score, 10)),
        })
        fig, ax = plt.subplots(figsize=(7, 3))

        sns.lineplot(data=curve_data.melt(id_vars=['AA']),
                     x='AA', y='value', color='#2c7fb8', ax=ax)
        ax2 = ax.twinx()
        # rec = self.mut_rec
        # rec[rec<self.mut_rec/10]=0
        # if sum(rec)==0:
            # return
        if self.mut_idx:
            plt.vlines(x=self.mut_idx-1, ymin=0,
                    ymax=self.mut_rec, colors='#dd1c77')
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='w', lw=4, label=self.name),
            Line2D([0], [0], color='#2c7fb8', lw=4, label='ES Score'),
        Line2D([0], [0], color='#dd1c77', lw=4, label='Mutations')]
        plt.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1.1))
        ax.set_ylabel('ES Score')
        ax.set_xlim(1, len(self.lddt)+1)
        ax.set_xlabel("")
        ax2.set_ylabel('Recurrence')
        if output:
            fig.savefig(output+"/"+self.name+'.png')
        plt.close()
        return


def abs_grad(f):
    return normalize(np.absolute(np.gradient(f)))


def square_grad(f):
    return normalize(np.gradient(f) ** 2)


def smooth(arr, kernel_size=10, method='gaussian'):
    if method == 'conv':
        kernel = np.ones(kernel_size) / kernel_size
        f = np.convolve(arr, kernel, mode='same')
    elif method == 'gaussian':
        f = gaussian_filter1d(arr, sigma=kernel_size/2)
    return f


def demax(x):
    return x/x.max()


def normalize(x):
    new_x = x-x.min()
    return new_x/new_x.max()


def normalize_reg(x):
    x = x - x[x.nonzero()].min()
    x = x / x.max()
    x[x < 0] = 1
    return 1-x


def window_std(x, window):
    x_ = np.concatenate([np.zeros(window)+x[1], x, np.zeros(window)+x[-1]])
    return np.array([x_[i:i+2*window+1].std() for i in range(x.size)])


def get_3d_avg(x, pairwise_interaction):
    x = x * pairwise_interaction
    x = x.sum(1)/((x > 0).sum(1)+0.01)
    return demax(x)

def get_3d_prod(x, pairwise_interaction):
    x = x * pairwise_interaction+1
    x = x.prod(1)
    return demax(x)

def get_3d_max(x, pairwise_interaction):
    x = x * pairwise_interaction
    x = x.max(1)
    return demax(x)

def get_3d_maxmin(x, pairwise_interaction):
    x = x * pairwise_interaction
    x = x.max(1) - x.min(1)
    return demax(x)

def main(arg):
    genename_to_uniprot = pd.read_csv(
        arg.uniprot, sep='\t').set_index('To').to_dict()['From']
    mut = pd.read_csv(arg.input+"/mutations.txt", header=None,
                      sep='\t', names=['recurrence', 'gene', 'pos'], na_filter=False)
    mut['pos'] = mut.pos.astype(int)
    oncogene = np.loadtxt("oncogene.txt", dtype=str)
    tsg = np.loadtxt("tsg.txt", dtype=str)
    lddt = dict()
    with open(arg.lddt, 'r') as f:
        for line in f:
            id, score = line.strip().split('\t')
            lddt[id] = np.array(score.split(",")).astype(float)

    if arg.plot and exists(arg.input + '/' + arg.chunk + '.scores.txt'):
        score_stats = pd.read_csv(arg.input + '/' + arg.chunk + '.scores.txt')
        score_stats['genetype'] = 'Other'
        score_stats.loc[score_stats.gene.isin(oncogene) & ~score_stats.gene.isin(tsg), 'genetype'] = 'Oncogene'
        score_stats.loc[score_stats.gene.isin(tsg) & ~score_stats.gene.isin(oncogene), 'genetype'] = 'TSG'
        score_stats.loc[score_stats.gene.isin(oncogene) & score_stats.gene.isin(tsg), 'genetype'] = 'Both'
        
        
        from statannotations.Annotator import Annotator
        fig, ax = plt.subplots(figsize=(9,5))   
        order = ["Hotspot", "Non hotspot","Not mutated"]
        ax = sns.violinplot(data=score_stats, x='mutation', y='scores', ax=ax, cut=0, order=order,palette='Set2', linewidth=2)
        
        pairs=[
                                ("Hotspot", "Non hotspot"),
                                ("Hotspot", "Not mutated"),
                                ("Not mutated", "Non hotspot"), 
                            ]
        annotator = Annotator(ax, pairs, data=score_stats, x='mutation', y='scores', order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', show_test_name=False)
        annotator.apply_test()
        ax, test_results = annotator.annotate()
        ax.set_xlabel('')
        ax.set(ylim=(0, 1))
        ax.set_ylabel('ES score distribution')
        # plt.legend(loc='upper right', title='')
        plt.savefig(arg.input+"/Comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    else:
        genes = np.loadtxt(arg.input+"/"+arg.chunk, dtype='str', ndmin=1)
        score_stats = []
        for i, g in tqdm(enumerate(genes)):
            print(g)
            # if i>10:
            #     break
            try:
                gene = Gene(g, lddt, genename_to_uniprot, mut, 
                arg.gap, 
                arg.interaction, 
                arg.kernel, 
                arg.smooth_method, 
                arg.hotspot,
                g in oncogene,
                g in tsg,
                True,
                arg.dimer,
                arg.transition,)
                        
            except:
                continue
            # if gene:
            # os.makedirs(arg.input+"/plots/", exist_ok=True)
            # if gene.oncogene or gene.tsg:
            #     gene.plot_curve(arg.input+"/plots/")
            
            score_stat = []
            if hasattr(gene, 'hotspot_idx'):
                score_stat.append(pd.DataFrame({
                    'scores': gene.hotspot_scores, 
                    'pos': gene.hotspot_pos,
                    'rec': gene.hotspot_rec,
                    'mutation': 'Hotspot',
                    'mutprob': gene.hotspot_mut_prob,
                    'esm': gene.hotspot_esm,
                    'gene': gene.name,
                    'oncogene': gene.oncogene,
                    'tsg': gene.tsg
                    }))
            if hasattr(gene, 'non_hotspot_idx'):
                score_stat.append(pd.DataFrame({
                    'scores': gene.non_hotspot_scores, 
                    'pos': gene.non_hotspot_pos,
                    'rec': gene.non_hotspot_rec,
                    'mutation': 'Non hotspot', 
                    'mutprob': gene.non_hotspot_mut_prob,
                    'esm': gene.non_hotspot_esm,
                    'gene': gene.name, 
                    'oncogene': gene.oncogene,
                    'tsg': gene.tsg
                    }))
            if hasattr(gene, 'not_mut_idx'):
                score_stat.append(pd.DataFrame({
                    'scores': gene.not_mut_scores, 
                    'pos': gene.not_mut_pos,
                    'rec': 0,
                    'mutation': 'Not mutated', 
                    'mutprob': gene.not_mut_mut_prob,
                    'esm': gene.not_mut_esm,
                    'gene': gene.name, 
                    'oncogene': gene.oncogene,
                    'tsg': gene.tsg
                    }))
            try:
                score_stat = pd.concat(score_stat, axis=0)
            except:
                continue
            score_stats.append(score_stat)
            gene = None     


        score_stats = pd.concat(score_stats, axis=0)
        score_stats['group']=arg.input
        score_stats.to_csv(arg.input+"/"+ arg.chunk + ".scores.txt")



    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'lddt', type=str, default="plddt/9606.pLDDT.tdt", help='AlphaFold pLDDT table file')
    parser.add_argument('uniprot', type=str, default="uniprot_to_genename.txt",
                        help='Uniprot to gene name mapping file')
    parser.add_argument('input', type=str, help='input dir')
    parser.add_argument('chunk', type=str, help='input chunk')
    parser.add_argument('--dimer', action='store_true', help='plot dimer')
    parser.add_argument('--plot', action='store_true', help='plot only')
    parser.add_argument('--gap', default=3, type=int, help='secondary structure gap')
    parser.add_argument('--smooth_method', type=str, default='conv', help='plot only')
    parser.add_argument('--interaction', default=15, type=int, help='AA interaction threshold in armstrong')
    parser.add_argument('--hotspot', default=0.1, type=float, help='hotspot definition')
    parser.add_argument('--kernel', default=10, type=int, help='smooth kernal size')
    parser.add_argument('--transition', type=str, help='path to mutation transition probability', default='ALL_aa_transition.csv')
    arg = parser.parse_args()
    if os.path.exists(arg.input+"/"+ arg.chunk + ".scores.txt"):
        print("Already done")
        arg.plot = True
    main(arg)

# %%
