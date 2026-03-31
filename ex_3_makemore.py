import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter

# Load sequences from FASTA file
records = list(SeqIO.parse("uniprot_sprot.fasta", "fasta"))
random.seed(420)
seq_list = [str(r.seq) for r in random.sample(records, 3000)]
print(seq_list[:2])

# baby function to calculate average sequence length
def avg_seq_length(sequences):
    return sum(len(seq) for seq in sequences) / len(sequences)

print(avg_seq_length(seq_list))
                
# Keep only canonical amino acids
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
seq_list = [s for s in seq_list if set(s).issubset(valid_aa)]

chars = sorted(list(set("".join(seq_list))))
stoi = {s:i+1 for i,s in enumerate(chars)}  # start from 1 for padding 0
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(stoi)

# Building dataset of blocks of 35 characters
block_size = 35

def build_dataset(seq):
  X, Y = [], []
  for aa in seq:

    #print(w)
    context = [0] * block_size # Padding out context window
    for ch in aa + '.':
      ix = stoi[ch] # Converting character to index
      X.append(context.copy()) # Storing context
      Y.append(ix) # Storing next character
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # Sliding context window forward by 1 character

  X = torch.tensor(X) # Converting contexts to tensor
  Y = torch.tensor(Y) # Converting next characters to tensor
  print(X.shape, Y.shape)
  return X, Y

# Building training, dev and test sets
import random
random.seed(42)
random.shuffle(seq_list)
n1 = int(0.8*len(seq_list))
n2 = int(0.9*len(seq_list)) # 80 10 10

Xtr, Ytr = build_dataset(seq_list[:n1])
Xdev, Ydev = build_dataset(seq_list[n1:n2])
Xte, Yte = build_dataset(seq_list[n2:])

# Creating initial embedding matrix
C = torch.randn((len(stoi), 2))

g = torch.Generator().manual_seed(123456) # for reproducibility

# Initialising model parameters
C = torch.randn((21, 10), generator=g) # Character embedding matrix - 21 characters by 10 embedding size
W1 = torch.randn((block_size * 10, 40), generator=g) # Concatenated embeddings (35*10) by 40 neurons - hidden layer weights
b1 = torch.randn(40, generator=g) # 40 neurons - hidden layer biases
W2 = torch.randn((40, 21), generator=g) # 40 neurons, 21 output layer size - hidden layer weights
b2 = torch.randn(21, generator=g) # 21 output layer size - hidden layer biases
parameters = [C, W1, b1, W2, b2]

# Gradient tracking
for p in parameters:
  p.requires_grad = True

# Log-scale range for learning rate search
lre = torch.linspace(-3, 0, 1000) # between 0.001 and 1

# Converting log scale to actual rates
lrs = 10**lre

# Lists for tracking training stats
lri = []
lossi = []
stepi = []
dev_lossi = []
dev_steps = [] 

for i in range(20000): # Iterations

    # Constructing minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # Forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    #print(loss.item())

    # Backward pass (resetting gradients first)
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    #lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # Printing loss every 1000 steps
    if i % 1000 == 0:
        print(f'step{i}: loss {loss.item():.4f}')
    
    # Evaluating on dev set every 1000 steps
    if i % 1000 == 0:
        with torch.no_grad():
            emb_dev = C[Xdev]
            h_dev = torch.tanh(emb_dev.view(-1, block_size*10) @ W1 + b1)
            logits_dev = h_dev @ W2 + b2
            loss_dev = F.cross_entropy(logits_dev, Ydev)

            dev_lossi.append(loss_dev.log10().item())  # Saving dev loss
            dev_steps.append(i)                        # Saving step number

    print(f"Step {i}: train loss {loss.item():.4f}, dev loss {loss_dev.item():.4f}")



    # Tracking stats for plotting
    #lri.append(lre[i])
    stepi.append(i) # check good learning rate?
    lossi.append(loss.log10().item())

print(loss.item())

# Plotting train vs dev loss to check for overfitting
plt.plot(stepi, lossi, label='Train loss')
plt.plot(dev_steps, dev_lossi, label='Dev loss')
plt.xlabel('Step')
plt.ylabel('log10(Loss)')
plt.legend()
plt.title('Train vs Dev Loss')
plt.show()

# Evaluating final loss on training set
emb = C[Xtr]
h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ytr)
print(loss)

# Evaluating final loss on dev set
emb = C[Xdev] 
h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)
logits = h @ W2 + b2 
loss = F.cross_entropy(logits, Ydev)
print(loss)

# Defining properties of amino acids 
hydrophobic_aromatic = set("FWY")
hydrophobic_aliphatic = set("AILMV")
polar_uncharged = set("CSTNQ")
charged_acidic = set("DE")
charged_basic = set("RHK")
special = set("GP.")

# Assigning colors based on amino acid properties
colors = []
for i in range(C.shape[0]):
    ch = itos[i]
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

# Visualising dimensions 0 and 1 of the embedding matrix for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200, c=colors)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# Initialising context window with padding tokens for sampling
context = [0] * block_size
C[torch.tensor([context])].shape

# Sampling from the model
g = torch.Generator().manual_seed(12345 + 10)

generated_sequences = []

for _ in range(1000):

    out = []
    context = [0] * block_size # Initialising with all ...
    while True:
      emb = C[torch.tensor([context])] 
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0: # Ending at full stop
        break

    generated_sequences.append(''.join(itos[i] for i in out)) # Decoding and printing generated sequences

# function to check frequencies of first amino acid of sequences
def first_aa(sequences):
    first_aas = [seq[0] for seq in sequences if len(seq) > 0]
    aa_counts = Counter(first_aas)
    sorted_counts = aa_counts.most_common()

    print("First amino acid counts (descending):")
    for aa, count in sorted_counts:
        print(f"{aa}: {count}")
    
    # % of most frequent amino acid
    if sorted_counts:
        top_aa, top_count = sorted_counts[0]
        percentage = (top_count / len(first_aas)) * 100
        print(f"\nMost common first amino acid: {top_aa} ({percentage:.2f}%)")

#print(generated_sequences)
print(avg_seq_length(generated_sequences))
first_aa(generated_sequences)
first_aa(seq_list)