# =============================================================================
# Neural oscillation trggered by time delay
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

#%% model 
class LIF:
    def __init__(self, N, gamma, eps, J, tau, theta, tau_rp, Vr):
        ## Neural structure 
        self.N = N
        self.N_E = int(N/(1+gamma))
        self.N_I = N - self.N_E 
        self.C_E = int(eps * self.N_E)
        self.C_I = int(eps * self.N_I)
        self.J = J
        
        ## Dynamics
        self.tau = tau
        self.theta = theta
        self.tau_rp = tau_rp
        self.Vr = Vr
        
     
    def initialize(self, g, delta, ratio, dt):
        # coupling matrix
        self.J_E = self.J  
        self.J_I = - g * self.J
        Jmat_E = np.zeros((self.N, self.N_E))
        Jmat_I = np.zeros((self.N, self.N_I))
        for i in range(self.N):
            act_E = np.random.permutation(self.N_E)[0:self.C_E]
            Jmat_E[i:i+1, act_E] = self.J_E
            act_I = np.random.permutation(self.N_I)[0:self.C_I]
            Jmat_I[i:i+1, act_I] = self.J_I
        self.Jmat = np.concatenate((Jmat_E, Jmat_I), axis=1)

        # dynamics
        self.delta = delta
        v_thr = self.theta / (self.J * self.C_E * self.tau)
        self.v_ext = v_thr * ratio
        self.dt = dt
        self.spike_matrix = np.zeros((self.N, int(delta/dt)))
        self.reset_matrix = np.zeros((self.N, int(self.tau_rp/dt)))
        
        
    def update_spike_mat(self, current_spike):
        self.spike_matrix = np.roll(self.spike_matrix, -1, axis=1)
        self.spike_matrix[:,-1] = current_spike.reshape(-1) * 1
    
    
    def update_reset_mat(self, current_spike):
        self.reset_matrix = np.roll(self.reset_matrix, -1, axis=1)
        self.reset_matrix[:,-1] = 0 
        self.reset_matrix[np.where((current_spike.reshape(-1))==1),:] = 1
        
    
    def run(self, T):
        steps = int(T/self.dt)
        V0 = np.random.uniform(10, 20, (self.N, 1))
        V_previous = V0 * 1
        V_traj = np.zeros([self.N, steps])
        for s in range(steps):
            # update the membrane potential
            V_current = (1 - self.dt/self.tau) * V_previous
            V_current += self.dt * np.dot(self.Jmat, self.spike_matrix[:,0:1])
            V_current += self.J_E * np.random.poisson(lam=self.v_ext*self.dt*self.C_E, size=[self.N, 1])
            V_current[np.where(self.reset_matrix[:,0]==1),0] = self.Vr
            
            # update the spike/reset matrix
            current_spike = np.heaviside(V_current - self.theta, 0)
            self.update_spike_mat(current_spike)
            self.update_reset_mat(current_spike)
            
            # record and set V
            V_traj[:,s:s+1] = V_current*1
            V_previous = V_current*1
            print("\rTime: {:.2f}/{:.2f}ms".format((s+1)*self.dt, T), end='')
        return np.array(V_traj)
            
            
        
#%% main
if __name__ == '__main__':

    # neural structure
    N = 12500
    gamma = 0.25
    eps = 0.1 
    J = 0.1
    tau = 20
    theta = 20
    tau_rp = 2 
    Vr = 10
    
        
    # run
    g = 6
    delta = 0.1
    ratio = 4
    dt = 0.1
    T = 200
    def run_LIF(delta, index):
        LIF1 = LIF(N, gamma, eps, J, tau, theta, tau_rp, Vr)   
        LIF1.initialize(g, delta, ratio, dt)
        V_traj = LIF1.run(T)
        Spike_traj = (np.heaviside(V_traj - theta, 0))
        np.save("Spike_traj_{:.2f}_{}.npy".format(delta, index), Spike_traj) 
    
    #%% run
    for d in [1.5]:
        for i in range(5):
            run_LIF(d, 4)


    #%% figure
    # data
    delta, index = 3, 0
    Spike_data = np.load("Spike_traj_{:.2f}_{}.npy".format(delta, index), allow_pickle=True)
    
    # set time range
    t1, t2 = 100, 200
    step_t1, step_t2 = int(t1/dt), int(t2/dt)
    Spike_traj = Spike_data[:, step_t1:step_t2]
    
    # plot
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    Spike_index = Spike_traj * (np.arange(1, N+1, 1)).reshape(-1,1)
    Spike_index = np.where(Spike_index==0, None, Spike_index)  
    for i in range(100):
        plt.plot(np.arange(step_t1, step_t2, 1), Spike_index[i],\
                 lw=0, marker='o', ms=0.5, c='k')
    plt.xlim(1000,2000)
    plt.xticks(np.arange(1000, 2100, 100), np.arange(100, 210, 10), size=15)
    plt.yticks(size=15)
    plt.ylabel("index", size=20)
    
    
    plt.subplot(4,1,3)
    Spike_hist = Spike_traj * (np.arange(step_t1, step_t2, 1)).reshape(1,-1)
    Spike_hist = Spike_hist[Spike_hist>0]
    plt.hist(Spike_hist, bins=int((t2-t1)/dt), color='k', alpha=0.8)
    plt.xlabel("time(ms)", size=20)
    plt.xlim(1000,2000)
    plt.xticks(np.arange(1000, 2100, 100), np.arange(100, 210, 10), size=15) 
    plt.yticks(size=15)
    plt.ylabel("firing rate", size=20)
    plt.subplots_adjust(hspace=0.2)
    plt.savefig("Spike_delta{:.2f}.png".format(delta), dpi=300, bbox_inches='tight')













































