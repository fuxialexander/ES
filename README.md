# ES score: unsupervised prediction of cancer hotspot using evolutionary and structural information


This should be rather fast as it's just visualizing all computed ES scores. (conda environment setup time depends on internet connection condition; the visualization time should be very minimal) 

```bash
git clone git@github.com:fuxialexander/ES.git
conda env create -f environment.yml
conda activate es
cd website
python app.py
```

To plot ES score for new genes, you can use the following command:
```
python plot.py --transition cosmic_aa_transition.csv  --gap 5 --interaction 15 --hotspot 0.1 --kernel 10 --smooth_method 'gaussian' plddt/9606.pLDDT.tdt uniprot_to_genename.txt {data_folder} genes.txt
```

where `9606.pLDDT.tdt` can be downloaded at https://github.com/normandavey/ProcessedAlphafold/blob/main/9606.pLDDT.tdt.zip
and {data_folder} contains two files: `genes.txt` (which list the genes you want to predict) and `mutations.txt` (which list mutated residue #, gene, and mutation frequency). 

Such files for COSMIC mutations and oncogenes can be found in https://github.com/fuxialexander/ES/tree/main/rank_all_cosmic.

The running result will be in a file like https://github.com/fuxialexander/ES/blob/main/rank_all_cosmic/genes.txt.scores.txt, which you can use the following command or https://github.com/fuxialexander/ES/blob/main/plot_gs_rank.py script to visualize

```
python plot.py --plot --transition cosmic_aa_transition.csv  --gap 5 --interaction 15 --hotspot 0.1 --kernel 10 --smooth_method 'gaussian' plddt/9606.pLDDT.tdt uniprot_to_genename.txt {data_folder} genes.txt
```
