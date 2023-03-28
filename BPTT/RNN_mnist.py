# -*- coding: utf-8 -*-
#%% import
import numpy as np

#%% 定义函数
def f(x):
    return np.maximum(0,x)
    
def df(x):
    return np.where(x > 0, 1, 0)

def g_clip(g):
    g = np.where(g>1,1,g)
    g = np.where(g<-1,-1,g)
    return g

#%% 定义RNN类
class RNN:
    def __init__(self, n_in, n_rec, n_out, tau_m=1):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m

        # Initialize weights:
        # 后一层在第一维度
        self.w_in = np.random.normal(0,1,[n_rec,n_in])/(n_rec**0.5*n_in**0.5)
        self.w_rec = np.random.normal(0,1/(n_rec**0.5*n_rec**0.5),[n_rec,n_rec])
        self.w_out = np.random.normal(0,1/(n_rec**0.5*n_out**0.5),[n_out,n_rec])
                
        # rmsprop
        self.RMSp_in = RMS_prop()
        self.RMSp_rec = RMS_prop()
        self.RMSp_out = RMS_prop()
    
    def identity(self):
        self.w_rec = np.eye(self.n_rec)
    
    def accuracy(self,y,y_):
        targets = y_.argmax(axis=0)
        predicts = y.argmax(axis=0)
        return np.sum(targets==predicts)/np.size(targets)
    
    def run_bptt(self, x, y_, h0, eta, learning=False, optimizer='SGD'):
        # feedforward
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]

        gw_in, gw_rec, gw_out = 0, 0, 0  # gradients of weights

        u = np.zeros((t_max, self.n_rec, b_size))  # input (feedforward plus recurrent)
        h = np.zeros((t_max, self.n_rec, b_size))  # time-dependent RNN activity vector
                    
        y = np.zeros((self.n_out,b_size))  # RNN output
        err = np.zeros((self.n_out,b_size))  # readout error     
        
        #forward
        u[0] = np.dot(self.w_rec, h0)+ np.dot(self.w_in, x[0])
        h[0] = h0 + (-h0 + f(u[0]))/self.tau_m
        for tt in range(t_max-1):
            u[tt+1] = np.dot(self.w_rec, h[tt])+ np.dot(self.w_in, x[tt+1])
            h[tt+1] = h[tt] + (-h[tt] + f(u[tt+1]))/self.tau_m
            
        y = np.dot(self.w_out, h[t_max-1])

        if not learning:
            return self.accuracy(y,y_)
        
        #softmax层
        y = y - np.array(y.max(axis=0),ndmin=2)
        exp_y = np.exp(y) 
        sumofexp = np.array(exp_y.sum(axis=0),ndmin=2)
        softmax = exp_y/sumofexp
        CE_k = (-y_*np.log(softmax+1e-10)).sum(axis=0)
        CE = np.sum(CE_k)/np.size(CE_k)

        #backpropagation
        err = softmax - y_     #dL/dy             
        z = np.zeros((t_max, self.n_rec, b_size))
        
        # t = T
        z[-1] = np.dot((self.w_out).T, err)
        for tt in range(t_max-1, 0, -1):
            z[tt-1] = z[tt]*(1 - 1/self.tau_m)
            z[tt-1] += (np.dot((self.w_rec).T,z[tt]*df(u[tt]) )/self.tau_m)            
            # Updates for the weights:
            gw_rec += 1/(self.tau_m)*np.dot(z[tt]*df(u[tt]),h[tt-1].T)/b_size
            gw_in += 1/(self.tau_m)*np.dot(z[tt]*df(u[tt]),x[tt].T)/b_size
        gw_rec += 1/(self.tau_m)*np.dot(z[0]*df(u[0]),h0.T)/b_size
        gw_in += 1/(self.tau_m)*np.dot(z[0]*df(u[0]),x[0].T)/b_size


        gw_out += 1*np.dot(err, h[t_max-1].T)/b_size

        '''
        gw_out = g_clip(gw_out)
        gw_rec = g_clip(gw_rec)
        gw_in  = g_clip(gw_in)
        '''

        if optimizer=='SGD':
            self.w_out = self.w_out - eta*gw_out
            self.w_rec = self.w_rec - eta*gw_rec
            self.w_in  = self.w_in  - eta*gw_in
        if optimizer=='RMS':
            self.w_out = self.RMSp_out.New_theta(self.w_out,gw_out,eta)
            self.w_rec = self.RMSp_rec.New_theta(self.w_rec,gw_rec,eta)
            self.w_in  = self.RMSp_in.New_theta(self.w_in,gw_in,eta)
        
        return CE,gw_out,gw_rec,gw_in

#%% 定义优化器
class RMS_prop:
    def __init__(self):
        self.lr=0.1
        self.beta=0.9
        self.epislon=1e-8
        self.s=0
        self.t=0
    
    def initial(self):
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.lr = eta
        self.t += 1
        g=gradient
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        theta -= self.lr*g/pow(self.s+self.epislon,0.5)
        return theta

#%% 加载数据函数
def dataload(data_list):
    dataFeatures=[]
    labels=[]
    for record in data_list:
        all_values=record.split(',')
        data = (np.asfarray(all_values[1:])/255.0)
        label = int(all_values[0])
        dataFeatures.append(data)
        labels.append(label)
    dataFeatures = np.array(dataFeatures)
    labels = np.array(labels,ndmin=2).T
    dataMat = np.concatenate((labels,dataFeatures),axis=1)
    return dataMat

#%% 训练和测试函数
def train(net, b_size, batches, train_data):
    #计算均值
    CEs, g_out, g_in, g_rec = 0, 0, 0, 0   
    #设置批次
    data_all = np.random.permutation(train_data)
    data = data_all[:,1:]
    targets = data_all[:,0]
    h_init = 0.1*np.ones([n_rec,b_size])
    for ii in range(batches):
        x = (data[ii*b_size:(ii+1)*b_size].T).reshape(28,28,b_size)
        targets_list = targets[ii*b_size:(ii+1)*b_size]
        y_ = (np.eye(10)[targets_list.astype(int)]).T
        CE,go,gr,gi  = net.run_bptt(x, y_,h_init, 
                       eta=learn_rate, learning=True, optimizer='RMS')      
        CEs += CE
        g_out +=  np.mean(np.abs(go))     
        g_rec +=  np.mean(np.abs(gr))    
        g_in +=  np.mean(np.abs(gi))    
    return CEs/batches, g_out/batches, g_rec/batches, g_in/batches

    
def test(net, test_data):  
    #计算总数
    accuracies = 0  
    #设置批次
    batches = 10
    b_size = 1000    
    data_all = np.random.permutation(test_data)
    data = data_all[:,1:]
    targets = data_all[:,0]
    h_test = 0.1*np.ones([n_rec,b_size])
    for ii in range(batches):      
        x_test = (data[ii*b_size:(ii+1)*b_size].T).reshape(28,28,b_size)
        targets_list = targets[ii*b_size:(ii+1)*b_size]
        y_test = (np.eye(10)[targets_list.astype(int)]).T
        accuracy = net.run_bptt(x_test,y_test,h_test,eta=learn_rate,learning=False)
        accuracies += accuracy
    return accuracies/batches            

#%%  主程序
#加载Minst数据
train_data_file=open("mnist_train.csv",'r')
train_data_list=train_data_file.readlines()
train_data_file.close()
train_data = dataload(train_data_list)

test_data_file=open("mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()
test_data = dataload(test_data_list)
print('Data is loaded.')    

#设置参数
batches = 100
b_size = 128
learn_rate = 1e-3
n_in, n_rec, n_out = 28, 150, 10

#%%
CEs = []
accuracies = []
g_out = []
g_rec = []
g_in  = []   
Total_epoch = 80
net1 = RNN(n_in, n_rec, n_out)
net1.identity()
for epoch in range(Total_epoch):       
    CE,go,gr,gi = train(net1, b_size, batches, train_data)    
    CEs.append(CE)
    g_out.append(go)
    g_rec.append(gr)
    g_in.append(gi)
    accuracy = test(net1, test_data)
    accuracies.append(accuracy)
    print('Epoch {}. CE = {}; Accuracy = {}'.format(epoch,CE,accuracy))
#%%
w = net1.w_rec
eigVals,eigVects=np.linalg.eig(np.mat(w))   
net2 = RNN(n_in, n_rec, n_out)
w2 = net2.w_rec
eigVals1,eigVects1=np.linalg.eig(np.mat(w2))    
import matplotlib.pyplot as plt
x1 = np.real(eigVals1)
y1 = np.imag(eigVals1)
x = np.real(eigVals)
y = np.imag(eigVals)
plt.figure(figsize=(10,8))
plt.scatter(x1,y1,c='r',s=5)
plt.scatter(x,y,c='b',s=20)    

    
#%%
def sample():
    #训练轨迹
    CEs = []
    accuracies = []
    g_out = []
    g_rec = []
    g_in  = []   
    Total_epoch = 80
    net1 = RNN(n_in, n_rec, n_out)
    net1.identity()
    for epoch in range(Total_epoch):       
        CE,go,gr,gi = train(net1, b_size, batches, train_data)    
        CEs.append(CE)
        g_out.append(go)
        g_rec.append(gr)
        g_in.append(gi)
        accuracy = test(net1, test_data)
        accuracies.append(accuracy)
        print('Epoch {}. CE = {}; Accuracy = {}'.format(epoch,CE,accuracy))
    return np.array(CEs), np.array(accuracies)
    
CE_5 = []
a_5 = []
for i in range(5):
    c,a = sample()
    CE_5.append(c)
    a_5.append(a)


CE_5 = np.array(CE_5)
a_5 = np.array(a_5)
error_m = np.mean(CE_5,axis=0)
error_std = np.std(CE_5,axis=0)
accuracy_m = np.mean(a_5,axis=0)
accuracy_std = np.std(a_5,axis=0)
import matplotlib.pyplot as plt
x = np.arange(1,81,1)
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx() 
label1, = ax1.plot(x, error_m, c='orangered',linewidth = 2)  
ax1.fill_between(x,error_m-error_std,error_m+error_std,color='orange',alpha = 0.2)      
label2, = ax2.plot(x, accuracy_m, c='blue',linewidth = 2)
ax2.fill_between(x,accuracy_m-accuracy_std,accuracy_m+accuracy_std,color='blue',alpha = 0.2) 
plt.legend([label1,label2], 
           ['$L_{CE}$','test accuracy'],
           loc = 1 , fontsize=20,ncol=2) 
ax1.set_ylim(-0.05, 2.1)
ax1.tick_params(labelsize=25)
ax2.set_ylim(0.45,1.1)
ax2.tick_params(labelsize=25)
ax1.set_xlabel('epoch',size=30) 
ax1.set_ylabel('training error',size=30)
ax2.set_ylabel('test accuracy', size=30)
plt.savefig('fig2.pdf',dpi=300, bbox_inches = 'tight')
























    
    
    
