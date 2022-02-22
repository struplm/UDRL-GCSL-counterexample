# UDRL-GCSL-counterexample
# Used CE (see get_CE() def. for details) posses just two transient states differing just in goal component (s=0,h=1,g=0) and (s=0,h=1,g=1)
# thus values and policies are parametrized just by CE goal component, s=0,h=1 is assumed implicitely


import numpy as np
import matplotlib.pyplot as plt


class Container() :
  pass

def get_CE(alpha) :
  """
  returns Command Extension
  alpha - parameter determining stochasticity of orig MDP, posible values are in (0.5,1]
          1 - deterministic environment
          0.5 - uniform transition distribution the most stochastic version of the environment
                (transitions are independent on actions)                 
  """
  CE = Container()
  CE.alpha = alpha
  CE.A = 2 # number of orig MDP actions (the same for CE)
  CE.S = 2 # number of orig MDP states
  CE.G = CE.S # number of goals
  CE.N = 1 # max rem. horizon
  CE.h0 = 1 # initial rem. horizon (fixed horizon problem -- initial horizon distribution is a singleton)
  CE.s0 = 0 # initial state (initial orig. MDP state distribution is a singleton)
  CE.pg0 = np.ones([CE.G]) # initial goal distribution is uniform
  CE.pg0 = CE.pg0 / np.sum(CE.pg0)
  # orig. MDP transition matrix
  CE.P = np.array([[alpha,1-alpha],[1-alpha,alpha]]) # orig. MDP transition kernel from state 0
                                                     # SxA matrix, indexed by next state and current action (s_{t+1}, a_t)
                                                     # the current state is always 0 (s_t = 0)
  CE.rho = np.array([0,1]) # identity map rho:S -> G(=S)
  return CE 


def get_opt(CE) :
  """
  Optimal values and policy
  threre are just two transient states differing just in goal component s=0,h=1,g=0 and s=0,h=1,g=1
  thus values and policies are parametrized just by CE goal component 
  """
  G = CE.G
  A = CE.A
  Q = np.zeros([G,A]) # "current state CE goal component" x action
  V = np.zeros([G])   # "current state CE goal component"
  pi = np.zeros([A,G]) # action x "current state CE goal component"
  for g in range(G) :
    for a in range(A) :
      Q[g,a] = CE.P[g,a]
    
    pi[:,g] = Q[g,:] == np.max(Q[g,:])
    pi[:,g] = pi[:,g]/np.sum(pi[:,g])
    V[g] = np.inner(Q[g,:],pi[:,g])
  opt = Container()
  opt.Q = Q
  opt.V = V
  opt.pi = pi
  opt.J = np.inner(CE.pg0,V) # GCSL objective
  return opt

def get_policy_values(CE,pi) :
  """
  return values for a policy pi
  threre are just two transient states differing just in goal component s=0,h=1,g=0 and s=0,h=1,g=1
  thus values and policies are parametrized just by CE goal component 
  """
  G = CE.G
  A = CE.A
  Q = np.zeros([G,A]) # "current state CE goal component" x action
  V = np.zeros([G])   # "current state CE goal component"
  for g in range(G) :
    for a in range(A) : 
      Q[g,a] = CE.P[g,a]
    V[g] = np.inner(Q[g,:],pi[:,g])
  J = np.inner(CE.pg0,V) # GCSL objective goal reaching objective
  return Q,V,J


def sample(dist) : # a sample from cathegorical dist
  return np.random.choice(len(dist),p=dist) 


def simulate_CE(traj_num,CE,pi) :
  """
  returns batch of traj_num rajectories of the CE
  """
  batch = np.zeros([traj_num,2,3+1],dtype=np.int) #traj_num x l(\tau) +1 x (num of CE components + 1)
  for traj_idx in range(traj_num) :
    # samle the initial CE stae
    s0 = CE.s0 #0
    h0 = CE.h0 #1
    g0 = sample(CE.pg0) # sample CE initial state goal component
    a0 = sample(pi[:,g0]) # sample first action    
    batch[traj_idx,0]=[s0,h0,g0,a0] 
    s1 = sample(CE.P[:,a0]) # sample the second (and final) CE original MDP state component
    h1 = h0-1
    g1 = g0
    batch[traj_idx,1]=[s1,h1,g1,-1]
  return batch
  
def UDRL(CE,pi0,it_num) :
  """
  Simulates it_num iterations of UDRL algorithm
  """
  G = CE.G
  A = CE.A
  V = np.zeros([it_num,G]) # it_num x current state CE goal component
  J = np.zeros([it_num]) # it_num
  pi = np.zeros([it_num,A,G]) # it_num x action x current state CE goal component
  traj_num = 10000 # number of trajectores to be simulated in each UDRL iteration
  
  pi[0] = pi0
  _,V[0],J[0] = get_policy_values(CE,pi0)
  for n in range(1,it_num) :
    batch = simulate_CE(traj_num,CE,pi[n-1])
    # because of fixed horizon 1 there is only posibility to choose a segment
    # from a trajectory: segment is always the whole trajectory
    # so sampling segments means just sampling tarjectories
    # Further a new policy is fitted to fit := p(a_0|s_0=0,l(sigma)=1,rho(s_1)) 
    fit = np.zeros([A,G])
    for sigma in batch :
      s0,h0,g0,a0 = sigma[0]
      s1,h1,g1,_  = sigma[1]      
      fit[a0,CE.rho[s1]] = fit[a0,CE.rho[s1]] +1
    fit[:,0] = fit[:,0]/np.sum(fit[:,0])
    fit[:,1] = fit[:,1]/np.sum(fit[:,1])
    pi[n] = fit
    _,V[n],J[n] = get_policy_values(CE,pi[n])
    res = Container()
    res.V = V
    res.J = J
    res.pi = pi
  return res


def RMSVE(V1,V2) :
  return np.sqrt(np.mean(np.square(V1-V2),axis=1))

def supnorm(pi1,pi2) :
  return np.max(np.abs(pi1-pi2),axis=(1,2))

def main() :
  it_num = 6
  
  ex0 = Container() # Experiment 0
  ex0.CE = get_CE(alpha = 1.0) # deterministic case
  A,G = ex0.CE.A,ex0.CE.G
  pi0 = np.ones([A,G])/A # uniform initial policy 
  ex0.res = UDRL(ex0.CE,pi0,it_num)
  ex0.opt = get_opt(ex0.CE)

  ex1 = Container() # Experiment 1
  ex1.CE = get_CE(alpha = 0.9)
  ex1.res = UDRL(ex1.CE,pi0,it_num)
  ex1.opt = get_opt(ex1.CE)

  ex2 = Container() # Experiment 2
  ex2.CE = get_CE(alpha = 0.6) # heavy stochasticity
  ex2.res = UDRL(ex2.CE,pi0,it_num)
  ex2.opt = get_opt(ex2.CE)
  #print(f"ex2.pi={ex2.pi}")


  ex3 = Container() # Experiment 3 -- is there a monotony in GCSL goal reaching objective J? 
  ex3.CE = get_CE(alpha = 0.6) # heavy stochasticity
  ex3.opt = get_opt(ex3.CE)
  pi0 = ex3.opt.pi # optimum initial condition
  ex3.res = UDRL(ex3.CE,pi0,it_num)  

  it_axis = np.arange(0,it_num)

  fig,ax = plt.subplots(1,1,figsize=(3,3*3/4),dpi=300) 
  ax.set_xlim(0,5)
  ax.set_ylim(0,0.55)
  ax.set_yticks([0, 0.5])
  ax.set_xticks([0,1, 5])
  ax.set_xlabel("iteration",labelpad = -10)
  ax.set_ylabel("RMSVE",labelpad = -12)  
  ax.plot(it_axis, RMSVE(ex0.res.V,ex0.opt.V),"ro-",label="$\\alpha = 1.0$")
  ax.plot(it_axis, RMSVE(ex1.res.V,ex1.opt.V),"go-",label="$\\alpha = 0.9$")
  ax.plot(it_axis, RMSVE(ex2.res.V,ex2.opt.V),"bo-",label="$\\alpha = 0.6$")  
  ax.legend(loc="upper right")  
  plt.savefig("RMSVE.png")

  fig,ax = plt.subplots(1,1,figsize=(3,3*3/4),dpi=300) 
  ax.set_xlim(0,5)
  ax.set_ylim(0,0.55)
  ax.set_yticks([0, 0.5])
  ax.set_xticks([0,1, 5])
  ax.set_xlabel("iteration",labelpad = -10)
  ax.set_ylabel("$||\\pi_n-\\pi^*||_{\infty}$",labelpad = -12)  
  ax.plot(it_axis, supnorm(ex0.res.pi,ex0.opt.pi),"ro-",label="$\\alpha = 1.0$")
  ax.plot(it_axis, supnorm(ex1.res.pi,ex1.opt.pi),"go-",label="$\\alpha = 0.9$")
  ax.plot(it_axis, supnorm(ex2.res.pi,ex2.opt.pi),"bo-",label="$\\alpha = 0.6$")  
  ax.legend(loc="center right")
  plt.savefig("supdist.png")

  fig,ax = plt.subplots(1,1,figsize=(3,3*3/4),dpi=300)
  ax.set_xlim(0,5)
  ax.set_ylim(0.5,0.61)
  ax.set_yticks([0.5, 0.6])
  ax.set_xticks([0,1, 5])
  ax.set_xlabel("iteration",labelpad = -10)
  ax.set_ylabel("$J(\pi_n)$",labelpad = -12)
  ax.plot([ it_axis[0],it_axis[-1] ], ex3.opt.J*np.ones([2]),"b--",label="$J(\pi^*)$")
  ax.plot(it_axis, ex3.res.J,"ro-",label="$\pi_0 = \pi^*$")
  ax.plot(it_axis, ex2.res.J,"go-",label="$\pi_0$ is uniform")
  ax.legend(loc="center right")
  plt.savefig("monotony.png")


if __name__ == '__main__':
  main()


