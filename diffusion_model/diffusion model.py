# =============================================================================
# An example of Diffusion model
# =============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.datasets import make_s_curve 


#%% load data
s_curve,_ = make_s_curve(10**4,noise=0.1) 
s_curve = s_curve[:,[0,2]]/10.0
data = s_curve.T

plt.figure(figsize=(5,5), dpi=200) 
plt.scatter(*data, color='steelblue', edgecolor='white')
plt.axis('on')
dataset = torch.Tensor(s_curve).float()


#%% transform parameter
# steps
num_steps = 100

# beta
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

# alpha related
alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)



#%% forward pass
# define xt via re-parametrization trick
def xt_f(x_0, t):
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)


num_shows = 10
fig, axs = plt.subplots(1, 11, figsize=(10,1), dpi=500)
for i in range(num_shows+1):
    if i == 0:
        x_t = dataset*1
    else:
        x_t = xt_f(dataset,torch.tensor([i*num_steps//num_shows-1]))
    axs[i].scatter(x_t[:,0], x_t[:,1], color='steelblue', s=5, linewidths=0.2, edgecolor='white')
    axs[i].set_axis_off()
    axs[i].set_title('$\mathbf{x}_{'+str(i*num_steps//num_shows)+'}$', color='k')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)


#%% backward pass
import torch
import torch.nn as nn
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        
        # MLP parameters
        self.linears = nn.ModuleList(
            [
                nn.Linear(2,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,num_units),
                nn.ReLU(),
                nn.Linear(num_units,2),
            ]
        )
        
        # embeddings vectors for time
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
        
    def forward(self, x, t):
        for idx,embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)       
        return x


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    
    # choose time_step to train    
    t = torch.randint(0, n_steps, size=(batch_size//2,))
    t = torch.cat([t, n_steps-1-t], dim=0)
    t = t.unsqueeze(-1)
    
    # compute xt
    e = torch.randn_like(x_0) 
    x = x_0*alphas_bar_sqrt[t] + e*one_minus_alphas_bar_sqrt[t]
    
    output = model(x,t.squeeze(-1))
    return (e - output).square().mean()


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    # given t
    t = torch.tensor([t])
    
    # sample from reconstructed p
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x,t)
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    # sample all the tracjory: from x[T] to x[0] 
    cur_x = torch.randn(shape)
    x_seq = [cur_x.detach().numpy()]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x.detach().numpy())
    return np.array(x_seq)


#%% Main Training
# initialization
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 10000 + 1

# MLP
model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

# training
for t in range(num_epoch):
    # mini-batch training
    for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        
    # callback    
    if (t%100==0):
        print("Epoch:{}, Loss={:.4f}".format(t, loss))
    if (t%400==0):
        x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)
        np.save("x_seq_{}.npy".format(t), x_seq)
        

#%% Figures
# backward
def plot(t):
    x_seqs = np.load("x_seq_{}.npy".format(t))
    fig, axs = plt.subplots(1, 11, figsize=(10,1), dpi=500)
    for i in range(11):
        x_t = x_seqs[i*10]
        axs[i].scatter(x_t[:,0], x_t[:,1], color='steelblue', s=5, linewidths=0.2, edgecolor='white')
        axs[i].set_axis_off()
        axs[i].set_title('$\mathbf{x}_{'+str(100-i*num_steps//num_shows)+'}$', color='k')
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

for t in np.arange(0, 10000, 400):
    plot(t)
