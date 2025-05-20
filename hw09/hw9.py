import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests

# 1. Setup and Data Preparation
URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
with open("names.txt", "w", encoding="utf-8") as f:
    f.write(requests.get(URL).text)

# Read and prepare the data
words = open('names.txt', 'r').read().splitlines()

# Build the vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# Build the dataset
block_size = 3
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X = torch.tensor(X)
Y = torch.tensor(Y)

# Split the data
n1 = int(0.8 * len(X))
n2 = int(0.9 * len(X))

# 2. Model Components
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

# 3. Model Initialization
n_embd = 24
n_hidden = 128

model = {
    'C': torch.randn((vocab_size, n_embd)),
    'h1': Linear(n_embd * block_size, n_hidden, bias=False),
    'bn1': BatchNorm1d(n_hidden),
    'tanh': Tanh(),
    'h2': Linear(n_hidden, vocab_size, bias=False),
    'bn2': BatchNorm1d(vocab_size),
}

with torch.no_grad():
    model['C'].uniform_(-0.1, 0.1)
    model['h1'].weight *= 0.2
    model['h2'].weight *= 0.1
    model['bn1'].gamma.fill_(1.0)
    model['bn1'].beta.fill_(0.0)
    model['bn2'].gamma.fill_(1.0)
    model['bn2'].beta.fill_(0.0)

parameters = [p for layer in model.values() for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

# 4. training
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(0, X.shape[0], (batch_size,))
    Xb, Yb = X[ix], Y[ix]

    emb = model['C'][Xb]
    x = emb.view(emb.shape[0], -1)
    x = model['h1'](x)
    x = model['bn1'](x)
    x = model['tanh'](x)
    x = model['h2'](x)
    x = model['bn2'](x)
    loss = F.cross_entropy(x, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

# 5. evaluation
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (X[:n1], Y[:n1]),
        'val': (X[n1:n2], Y[n1:n2]),
        'test': (X[n2:], Y[n2:]),
    }[split]
    emb = model['C'][x]
    x = emb.view(emb.shape[0], -1)
    x = model['h1'](x)
    x = model['bn1'](x)
    x = model['tanh'](x)
    x = model['h2'](x)
    x = model['bn2'](x)
    loss = F.cross_entropy(x, y)
    print(f"{split} loss: {loss.item():.4f}")

split_loss('train')
split_loss('val')
split_loss('test')

# 6. generating names
@torch.no_grad()
def generate():
    out = []
    context = [0] * block_size
    while True:
        emb = model['C'][torch.tensor([context])]
        x = emb.view(1, -1)
        x = model['h1'](x)
        x = model['bn1'](x)
        x = model['tanh'](x)
        x = model['h2'](x)
        x = model['bn2'](x)
        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    return ''.join(out)

print("\nGenerated names:")
for _ in range(10):
    print(generate())

# 7. 
print("\nInitialization analysis:")
uniform_loss = torch.log(torch.tensor(vocab_size)).item()
print(f"Theoretical uniform loss: {uniform_loss:.4f}")

@torch.no_grad()
def check_initial_loss():
    emb = model['C'][X[:32]]
    x = emb.view(32, -1)
    x = model['h1'](x)
    x = model['bn1'](x)
    x = model['tanh'](x)
    x = model['h2'](x)
    x = model['bn2'](x)
    return F.cross_entropy(x, Y[:32]).item()

print(f"Actual initial loss: {check_initial_loss():.4f}")
