# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:19:42 2020

@author: Vincent
"""


import numpy as np
#%% 数据载入
def dataload(traning_list, testing_list):
    data_list = np.concatenate((traning_list, testing_list),axis=0)
    dataFeatures=[]
    labels=[]
    for record in data_list:
        all_values=record.split(',')
        data = (np.asfarray(all_values[1:])/255.0)
        label = int(all_values[0])
        dataFeatures.append(data)
        labels.append(label)
    dataFeatures = np.mat(dataFeatures)
    labels = np.mat(labels)
    labels = np.transpose(labels)
    dataMat = np.concatenate((labels,dataFeatures),axis=1)
    dataMat = np.array(dataMat)
    return dataMat

training_data_file=open("mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()
test_data_file=open("mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()  
datalist = dataload(training_data_list, test_data_list)
train_data = datalist[0:60000]
test_data = datalist[60000:70000]


# 二值化
def binary_pattern(data):
    data[data>=0.5] = 1
    data[data<0.5] = -1
    return data

train_binary = binary_pattern(train_data)
test_binary = binary_pattern(test_data)



#%%% 定义DBN
class DBN:
    def __init__(self,structure):
        depth = len(structure)
        self.s = np.array(structure)
        self.adam = {}
        para = {}
        for i in range(depth-1):
            para['{}->{}'.format(i,i+1)] = {}
            para['{}->{}'.format(i,i+1)]['w'] = \
                np.random.randn(structure[i],structure[i+1])/(np.sqrt(structure[i]*structure[i+1]))
            para['{}->{}'.format(i,i+1)]['bv'] = \
                np.random.randn(structure[i],1)/np.sqrt(structure[i])
            para['{}->{}'.format(i,i+1)]['bh'] = \
                np.random.randn(structure[i+1],1)/np.sqrt(structure[i])
            self.adam['{}->{}'.format(i,i+1)] = {}
            self.adam['{}->{}'.format(i,i+1)]['w'] = Adam_train()
            self.adam['{}->{}'.format(i,i+1)]['bv'] = Adam_train()
            self.adam['{}->{}'.format(i,i+1)]['bh'] = Adam_train()
        self.para_dict = para
    
     
    # 变量取1的条件概率,alpha为外场
    def initial(self,layer):
        s = self.s
        if layer == '1->2':
            self.para_dict[layer]['w'] = np.random.randn(s[1],s[2])/np.sqrt(s[1]*s[2])
            self.para_dict[layer]['bv'] = np.random.randn(s[1],1)/np.sqrt(s[1])
            self.para_dict[layer]['bh'] = np.random.randn(s[1],1)/np.sqrt(s[1])
        if layer == '2->3':
            self.para_dict[layer]['w'] = np.random.randn(s[2],s[3])/np.sqrt(s[2]*s[3])
            self.para_dict[layer]['bv'] = np.random.randn(s[2],1)/np.sqrt(s[2])
            self.para_dict[layer]['bh'] = np.random.randn(s[2],1)/np.sqrt(s[2])
    
    def p_codition_binary(self,alpha):        
        return 1/(1+np.exp(-2*alpha))
    
    def hidden_unit(self,v_data,layer,mean_field=False):
        w = self.para_dict[layer]['w']
        bh = self.para_dict[layer]['bh']          
        if not mean_field:       
            alpha_h = np.dot(w.T,v_data) + bh
            flag_h = self.p_codition_binary(alpha_h)-np.random.uniform(0,1,alpha_h.shape) 
            h = self.sign(flag_h)
        else:
            h = self.mean_hidden(v_data, w, bh)
        return h
    
    def sample_v_pconti(self,alpha):
        x = np.random.uniform(0,1,alpha.shape)
        y = 2*x*np.sinh(alpha) + np.exp(-alpha)
        v = 1/alpha*np.log(y)
        return v
    
    def alpha_clip(self,alpha):
        alpha[alpha<-80] = -80
        alpha[alpha> 80] = 80
        return alpha
    
    def mean_hidden(self,v,w,bh):
        alpha_h = np.dot(w.T,v) + bh
        m_hidden_activity = np.tanh(alpha_h)
        return m_hidden_activity
    
    def sign(self,x):
        x[x>=0] = 1
        x[x<0] = -1
        return x
    
    def cd_binary(self,trial_data,k,layer,lr,gamma, optimizer='SGD',mean_field=False):
        '''
        ----
        parameter:
        tiral_data: shape[Nv,b_size]
        k: cd-k
        ----
        '''
        b_size = trial_data.shape[1]
        w = self.para_dict[layer]['w']
        bv = self.para_dict[layer]['bv']
        bh = self.para_dict[layer]['bh']        
        v_data = trial_data * 1
        h_data = self.mean_hidden(v_data, w, bh)
        
        v = v_data*1     
        for t in range(k):    
            alpha_h = np.dot(w.T,v) + bh
            flag_h = self.p_codition_binary(alpha_h)-np.random.uniform(0,1,alpha_h.shape)
            h = self.sign(flag_h)
            alpha_v = np.dot(w,h) + bv
            if not mean_field:
                flag_v = self.p_codition_binary(alpha_v)-np.random.uniform(0,1,alpha_v.shape)
                v = self.sign(flag_v)    
            else:
                v = np.tanh(alpha_v)
        v_model = v*1
        h_model = self.mean_hidden(v_model, w, bh)
        
        g_w = 1/b_size*np.dot(v_data,h_data.T) - 1/b_size*np.dot(v_model,h_model.T) - gamma*w
        g_bv = np.array(v_data.mean(1),ndmin=2).T - np.array(v_model.mean(1),ndmin=2).T
        g_bh = np.array(h_data.mean(1),ndmin=2).T - np.array(h_model.mean(1),ndmin=2).T

        if optimizer == 'SGD':
            w += lr*g_w
            bv += lr*g_bv
            bh += lr*g_bh
        if optimizer == 'adam':
            w = self.adam[layer]['w'].New_theta(w,g_w,lr)
            bv = self.adam[layer]['bv'].New_theta(bv,g_bv,lr)
            bh = self.adam[layer]['bh'].New_theta(bh,g_bh,lr)
          
        error = 1-np.mean((v_data==v_model))
        L2 = np.mean(np.sqrt(np.sum((v_data-v_model)**2,axis=0)))
        return error,L2
        
    def cd_continuous(self,trial_data,k,layer,lr,gamma):
        b_size = trial_data.shape[1]
        w = self.para_dict[layer]['w']
        bv = self.para_dict[layer]['bv']
        bh = self.para_dict[layer]['bh']         
        v_data = trial_data * 1
        h_data = self.mean_hidden(v_data, w, bh)
        v = v_data*1     
        for t in range(k):    
            alpha_h = np.dot(w.T,v) + bh
            alpha_h = self.alpha_clip(alpha_h)
            flag_h = self.p_codition_binary(alpha_h)-np.random.uniform(0,1,alpha_h.shape)
            h = self.sign(flag_h)
            alpha_v = np.dot(w,h) + bv
            alpha_v = self.alpha_clip(alpha_v)
            v = self.sample_v_pconti(alpha_v)
        v_model = v*1
        h_model = self.mean_hidden(v_model, w, bh)
        
        g_w = 1/b_size*np.dot(v_data,h_data.T) - 1/b_size*np.dot(v_model,h_model.T) - gamma*w
        g_bv = np.array(v_data.mean(1),ndmin=2).T - np.array(v_model.mean(1),ndmin=2).T 
        g_bh = np.array(h_data.mean(1),ndmin=2).T - np.array(h_model.mean(1),ndmin=2).T 

        w += lr*g_w
        bv += lr*g_bv
        bh += lr*g_bh

        error = np.mean(np.abs((v_data - v_model)/v_data))
        L2 = np.mean((v_data-v_model)**2)
        return error,L2

class Adam_train:
    def __init__(self):
        self.lr=0.3
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=0
        self.s=0
        self.t=0
    
    def initial(self):
        self.m = 0
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,lr):
        self.t += 1
        self.lr = lr
        g=gradient
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.s = self.beta2*self.s + (1-self.beta2)*(g*g)
        self.mhat = self.m/(1-self.beta1**self.t)
        self.shat = self.s/(1-self.beta2**self.t)
        theta += self.lr*self.mhat/(pow(self.shat,0.5)+self.epislon)
        return theta
       

#%% 逐层训练
structure = [784,200,200,200]
DBN1 = DBN(structure)
patterns = test_data[:,1:]
b_size = 100
total_b = 100
k = 1
gamma = 1e-5
continuous = False
mean_field = True

def training_RBM_ep(layer,patterns,k,b_size,total_b,lr,continuous = False,mean_field=False):
    patterns = np.random.permutation(patterns)
    error = 0
    for batch in range(total_b):
        trial_data = (patterns[batch*b_size:(batch+1)*b_size]).T
        if continuous:
            e,L2 = DBN1.cd_continuous(trial_data, k, layer, lr, gamma)
        else:
            e,L2 = DBN1.cd_binary(trial_data,k,layer,lr,gamma,optimizer='SGD',mean_field=mean_field)
        error += e
        print('\rRunining: {:.2f}%'.format((batch+1)/total_b*100),end='')
    error /= total_b
    return error,L2

#%% 第一层
errors0 = []
lr = 0.04
for ep in range(200): 
    if (ep+1)%100==0:
        lr = lr/2
    error,_ = training_RBM_ep('0->1', patterns,k,b_size,total_b,lr,continuous = False,mean_field=False)
    print('\nEpoch: {}, error: {:.4f}%'.format(ep,error*100))
    errors0.append(error)

#%% 第二层
patterns1 = (DBN1.hidden_unit(patterns.T,'0->1',mean_field=mean_field)).T
DBN1.initial('1->2')
errors1 = []
L2_1 = []
lr = 0.04
for ep in range(200):   
    if (ep+1)%100==0:
        lr = lr/2
    error,L2 = training_RBM_ep('1->2', patterns1,k,b_size,total_b,lr,mean_field=mean_field)
    L2_1.append(L2)    
    errors1.append(error)
    print('\nEpoch: {}, L2: {:.4f}'.format(ep,L2))

#%% 第三层
patterns2 = (DBN1.hidden_unit(patterns1.T,'1->2',mean_field=mean_field)).T
DBN1.initial('2->3')
errors2 = []
L2_2 = []
lr = 0.04
for ep in range(200):   
    if (ep+1)%100==0:
        lr = lr/2    
    error,L2 = training_RBM_ep('2->3', patterns2,k,b_size,total_b,lr,mean_field=mean_field)
    L2_2.append(L2)
    errors2.append(error)
    print('\nEpoch: {}, L2: {:.4f}'.format(ep,L2))
    
#%%
# 数据保存

    
#%% 
# 演化图
error0 = np.array(errors0)
L2_1 = np.array(L2_1)
L2_2 = np.array(L2_2)

import matplotlib.pylab as plt
plt.figure(figsize=(15,12))
plt.subplot(3,1,1)
epochs = np.arange(1,201,1)
plt.plot(epochs,errors0,lw=5,c='royalblue',label='Layer 0-->1: Discrete')
plt.text(50,0.10,'Error min={:.2f}%'.format(error0.min()*100),fontsize=30, va='top',color='k')
plt.xticks((0,50,100,150,200),fontsize=30)
plt.yticks((0.06,0.10,0.14,0.18),fontsize=30)
plt.legend(fontsize=30)
plt.ylabel('Error',size=30)

plt.subplot(3,1,2)
epochs = np.arange(1,201,1)
plt.plot(epochs,L2_1,lw=5,c='forestgreen',label='Layer 1-->2: Continuous')
plt.text(50,6,'L2-norm min={:.2f}'.format(L2_1.min()),fontsize=30, va='top',color='k')
plt.xticks((0,50,100,150,200),fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.ylabel('L2-norm',size=30)

plt.subplot(3,1,3)
epochs = np.arange(1,201,1)
plt.plot(epochs,L2_2,lw=5,c='sandybrown',label='Layer 2-->3: Continuous')
plt.text(50,7,'L2-norm min={:.2f}'.format(L2_2.min()),fontsize=30, va='top',color='k')
plt.xticks((0,50,100,150,200),fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=30)
plt.ylabel('L2-norm',size=30)
plt.xlabel('Epoch',size=30)
plt.savefig('Training Trajectories.png',dpi=300,bbox_inches = 'tight')

#%%
# 重构图片与误差误差
total_layer = 3  
def reconstruct(total_layer):
    v = patterns.T
    for i in range(total_layer):
        layer = '{}->{}'.format(i,i+1)
        w = DBN1.para_dict[layer]['w']
        bv = DBN1.para_dict[layer]['bv']
        bh = DBN1.para_dict[layer]['bh']         
        alpha_h = np.dot(w.T,v) + bh
        flag_h = DBN1.p_codition_binary(alpha_h)-np.random.uniform(0,1,alpha_h.shape)
        h = DBN1.sign(flag_h)        
        v = h*1
    for i in np.arange(total_layer-1,-1,-1):
        layer = '{}->{}'.format(i,i+1)
        w = DBN1.para_dict[layer]['w']
        bv = DBN1.para_dict[layer]['bv']
        bh = DBN1.para_dict[layer]['bh']  
        alpha_v = np.dot(w,h) + bv
        flag_v = DBN1.p_codition_binary(alpha_v)-np.random.uniform(0,1,alpha_v.shape)
        v = DBN1.sign(flag_v)  
        h = v*1
    return 1- np.mean((v==patterns.T)), v.T

error3, recon3 = reconstruct(3)
error2, recon2 = reconstruct(2)
error1, recon1 = reconstruct(1)

#%%
plt.figure(figsize=(20,4))
pattern = test_data[0,1:].reshape(28,28)
plt.subplot(141)
plt.title('Origin',size=30)
plt.imshow(pattern,cmap='gray')
plt.axis('off') 

pattern = recon1[0].reshape(28,28)
plt.subplot(142)
plt.title('Reconstruct:0-1-0\nError: {:.2f}%'.format(error1*100),size=20)
plt.imshow(pattern,cmap='gray')
plt.axis('off') 

pattern = recon2[0].reshape(28,28)
plt.subplot(143)
plt.title('Reconstruct:0-1-2-1-0\nError: {:.2f}%'.format(error2*100),size=20)
plt.imshow(pattern,cmap='gray')
plt.axis('off') 

pattern = recon3[0].reshape(28,28)
plt.subplot(144)
plt.title('Reconstruct:0-1-2-3-2-1\nError: {:.2f}%'.format(error3*100),size=20)
plt.imshow(pattern,cmap='gray')
plt.axis('off') 
plt.savefig('Recons.png',dpi=200,bbox_inches = 'tight')


#%%
#感受野
plt.figure(figsize=(20,4))
x = [2,28,81,193]

plt.subplot(141)
w = (DBN1.para_dict['0->1']['w'][:,x[0]]).reshape(28,28)
plt.axis('on') 
plt.imshow(w,cmap='viridis')
plt.colorbar(shrink=0.83)

plt.subplot(142)
w = (DBN1.para_dict['0->1']['w'][:,x[1]]).reshape(28,28)
plt.axis('on') 
plt.imshow(w,cmap='viridis')
plt.colorbar(shrink=0.83)

plt.subplot(143)
w = (DBN1.para_dict['0->1']['w'][:,x[2]]).reshape(28,28)
plt.axis('on') 
plt.imshow(w,cmap='viridis')
plt.colorbar(shrink=0.83)

plt.subplot(144)
w = (DBN1.para_dict['0->1']['w'][:,x[3]]).reshape(28,28)
plt.axis('on') 
plt.imshow(w,cmap='viridis')
plt.colorbar(shrink=0.83)
plt.savefig('RF.png',dpi=200,bbox_inches = 'tight')
