import random
import torch
import esm
from Bio import SeqIO
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare a smaller FASTA file with a random subset of 60 sequences - any more and it wouldn't fit in memory
full_file = "uniprot_sprot.fasta"
small_file = "uniprot_sprot_60.fasta"

random.seed(420)

records = list(SeqIO.parse(full_file, "fasta"))
subset = random.sample(records, 60)
print(subset[:2])

with open(small_file, "w") as out_f:
    SeqIO.write(subset, out_f, "fasta")

# Load sequences and prepare them for the model
sequences = []
for i, record in enumerate(SeqIO.parse(small_file, "fasta")):
    sequences.append((f"protein{i+1}", str(record.seq)))

batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

# Run the model to get embeddings 
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    # embeddings from the last layer
    embeddings = results["representations"][6]

# Concatenate all embeddings into a single matrix for PCA
all_embeddings = torch.cat([embeddings[i, 1:len(sequences[i][1])+1] for i in range(len(sequences))], dim=0)
all_embeddings = all_embeddings.cpu().numpy()  # Convert to numpy

# Get all amino acids in the dataset
unique_aas = sorted(list(set("".join(seq for name, seq in sequences))))
aa_embeddings = {aa: [] for aa in unique_aas}

# Collect embeddings
for i, (name, seq) in enumerate(sequences):
    seq_emb = embeddings[i, 1:len(seq)+1]  # skip start token
    for j, aa in enumerate(seq):
        if aa in aa_embeddings:
            aa_embeddings[aa].append(seq_emb[j].cpu().numpy())

# Compute mean embedding per amino acid
mean_embeddings = []
for aa in unique_aas:
    mean_embeddings.append(np.mean(aa_embeddings[aa], axis=0))
mean_embeddings = np.array(mean_embeddings)

# Define amino acid properties
hydrophobic_aromatic = set("FWY")
hydrophobic_aliphatic = set("AILMV")
polar_uncharged = set("CSTNQ")
charged_acidic = set("DE")
charged_basic = set("RHK")
special = set("GP.")

# Assign colors based on amino acid properties
colors = []
for ch in unique_aas:
    if ch in hydrophobic_aromatic:
        colors.append("red")
    elif ch in hydrophobic_aliphatic:
        colors.append("orange")
    elif ch in polar_uncharged:
        colors.append("blue")
    elif ch in charged_acidic:
        colors.append("green")
    elif ch in charged_basic:
        colors.append("yellowgreen")
    else:
        colors.append("gray")

# Perform PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(mean_embeddings)

# Plot
plt.figure(figsize=(8,8))
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=200, c=colors)
for i in range(emb_2d.shape[0]):
    plt.text(emb_2d[i,0], emb_2d[i,1], unique_aas[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()
