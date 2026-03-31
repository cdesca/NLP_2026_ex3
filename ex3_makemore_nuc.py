import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from Bio import SeqIO


records = list(SeqIO.parse("/Users/carlaescalante/UPF/Hivern 2026/Processament de Llenguatge Natural/ex3/ncbi_dataset/ncbi_dataset/data/GCF_000005845.2/cds_from_genomic.fna", "fasta"))
seqs = [str(r.seq) for r in records if len(str(r.seq)) % 3 == 0]
len(seqs)

def to_codon(seq):
    return [seq[i:i+3] for i in range(0, len(seq), 3)]


# function to check validity of sequences
def check_validity(sequence):
    # remove trailing '.' from all sequences
    sequence = [seq[:-1] if seq.endswith('.') else seq for seq in sequence if seq]

    # 1 - calculate average sequence length
    avg_length = sum(len(seq) for seq in sequence) / len(sequence)
    print(f"Average sequence length: {avg_length:.2f} nucleotides")

    # check reading frame: sequences should be multiples of 3
    count = sum(1 for seq in sequence if len(seq) % 3 == 0)
    print(f"Percent valid length: {count / len(sequence) * 100:.3f}%")

    # 3 - convert sequences to list of codons
    codon_seq = [to_codon(seq) for seq in sequence]

    # 4 - check start and stop codons
    count_ATG = 0
    count_start = 0
    count_stop = 0
    count_prem_stop = 0
    for seq in codon_seq: 
        # start codons
        if seq[0] != 'ATG':
            count_ATG += 1
        if seq[0] not in ('ATG', 'GTG', 'TTG'):
            count_start += 1

        # stop codons
        if seq[-1] not in ('TAA', 'TAG', 'TGA'):
            count_stop += 1
        if any(codon in ('TAA', 'TAG', 'TGA') for codon in seq[:-1]):
            count_prem_stop += 1

    # report metrics
    print(f"Percent not ATG: {count_ATG / len(codon_seq) * 100:.3f}%")
    print(f"Percent valid start: {(1 - count_start / len(codon_seq)) * 100:.3f}%")
    print(f"Percent valid stop: {(1 - count_stop / len(codon_seq)) * 100:.3f}%")
    print(f"Percent pre-stop: {count_prem_stop / len(codon_seq) * 100:.3f}%")


bases = sorted(list(set(''.join(seqs))))
stoi = {s:i+1 for i,s in enumerate(bases)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# Building dataset of blocks of 50 characters
block_size = 93

def build_dataset(seq):
  X, Y = [], []
  for nuc in seq:

    #print(w)
    context = [0] * block_size # Padding out context window
    for base in nuc + '.':
      ix = stoi[base] # Converting character to index
      X.append(context.copy()) # Storing 36-character context
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
random.shuffle(seqs)
n1 = int(0.8*len(seqs))
n2 = int(0.9*len(seqs)) # 80 10 10

Xtr, Ytr = build_dataset(seqs[:n1])
Xdev, Ydev = build_dataset(seqs[n1:n2])
Xte, Yte = build_dataset(seqs[n2:])

# Creating initial embedding matrix
C = torch.randn((len(stoi), 2))

g = torch.Generator().manual_seed(123456) # for reproducibility

# Initialising model parameters
C = torch.randn((5, 10), generator=g) # Character embedding matrix - 5 characters by 10 embedding size
W1 = torch.randn((block_size * 10, 40), generator=g) # Concatenated embeddings (3*10) by 40 neurons - hidden layer weights
b1 = torch.randn(40, generator=g) # 40 neurons - hidden layer biases
W2 = torch.randn((40, 5), generator=g) # 40 neurons, 5 output layer size - hidden layer weights
b2 = torch.randn(5, generator=g) # 5 output layer size - hidden layer biases
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

for i in range(30000): # Iterations

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

# Visualising dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
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

print(generated_sequences)

check_validity(generated_sequences)