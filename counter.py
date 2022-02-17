# UDRL-GCSL-counterexample
# Command Extension used in counterexample 


import numpy as np


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
  CE.A = 2 # number of orig MDP actions (the same for CE)
  CE.S = 2 # number of orig MDP states
  CE.G = S # number of goals
  CE.N = 1 # max rem. horizon
  CE.h0 = 1 # initial rem. horizon (fixed horizon problem -- initial horizon distribution is a singleton)
  CE.s0 = 0 # initial state
  CE.pg0 = np.ones([CE.G]) # initial goal distribution
  CE.pg0 = CE.pg0 / np.sum(CE.pg0)
  # orig. MDP transition matrix
  CE.P = np.array([[alpha,1-alpha],[1-alpha,alpha]) # SxA transition matrix, indexed by next state and current action (s_{t+1}, a_t)
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
      Q[g,a] = P[g,a]
    
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
      Q[g,a] = P[g,a]
    V[g] = np.inner(Q[g,:],pi[:,g])
  J = np.inner(CE.pg0,V) # GCSL objective
  return Q,V,J



def simulate_CE(traj_num,CE,pi)
  """
  returns batch of traj_num rajectories of the environment
  """
  batch = np.zeros([traj_num,2,3+1]) #traj_num x l(\tau) +1 x (num of CE components + 1)
  for traj_idx in range(raj_num) :
    # samle the initial CE stae
    s = CE.s0 #0
    h = CE.h0 #1
    g = sample(CE.pg)
    a = sample(pi[:,g])
    batch[traj_idx,0]=[s,h,g,a]
    s_ = sample(CE.P[:,a])
    h_ = h-1
    g_ = g
    batch[traj_idx,1]=[s_,h_,g_,np.nan]
  return batch
  
def UDRL(CE,pi0,it_num) :
  """
  Simulates it_num iterations of UDRL algorithm
  """
  V = np.zeros([it_num,G]) # it_num x current state CE goal component
  J = np.zeros([it_num]) # it_num
  pi = np.zeros([it_num,A,G]) # it_num x action x current state CE goal component
  
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
  return V,J,pi

def main() :
  it_num = 20
  CE = get_CE(alpha = 0.6)
  pi0 = np.ones([CE.A,CE.G])/CE.A # uniform initial policy 
  V,J,_ = UDRL(CE,pi0,it_num)
  opt = get_opt(CE)

  it_axis = np.arange(0,it_num)
  plt.figure()
  plt.plot(it_axis, RMSVE(V,opt.V))
  plt.show()
  
    


