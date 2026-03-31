# NLP_2026_ex3

## Dependencies: 

The code requires the following Python libraries:

| Library | Purpose |
|---------|---------|
| `pandas` | Loading and manipulating dataset of names |
| `numpy` | Numerical operations and arrays |
| `torch` | Building and training neural network |
| `torch.nn.functional` | NN functions (cross-entropy loss and softmax) |
| `matplitlib` | Visualizing training loss, validation loss, and character embeddings |
| `Bio` | Loading and parsing sequence data in FASTA format |
| `collections` | Counting and managing sequence data (Counter) |
| `sklearn` | PCA for embeddings |
| `itertools` | To create codon combinations |
| `ESM` | Specifically for the ESM protein language models |



### Additional Required Resources:

1. **Protein sequence files:**
Protein sequences used in this project were downloaded from UniProtKB at:
https://www.uniprot.org/uniprotkb
2. **DNA sequenc files:**
Coding DNA sequences for E. coli were obtained from NCBI RefSeq:
Download from: [https://www.ncbi.nlm.nih.gov/refseq/](https://www.ncbi.nlm.nih.gov/datasets/genome/?taxon=562)
RefSeq: GCF_000005845.2
Action -> Download -> Genomic coding sequences (FASTA)

Datasets should be in .fasta or .fna format.
