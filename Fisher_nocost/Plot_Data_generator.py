import gym
import matplotlib as mpl
import numpy as np
import matplotlib.gridspec as gridspec
from cycler import cycler
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO2
from fisher_env import FisherEnv
from stable_baselines.common import make_vec_env
import Plot_Tools as pltools
import Plot_Tools2 as plt_2
import time as time

start=time.time()
#load your model here
model=PPO2.load("./Fisher_tests_BayesRK4_ci_physics/feedBayes_steadyTrue_lro0.00025_ts30.0M_N4_ec0.001_0.45_1e5_theta0.1_eta.1_Custom6464_1e-3_RK4_nonorm_full_inf_s2_q0/rl_model_29900000_steps.zip")
feedback='Bayes'#'Markov' and 'Bayes are the choices

det=False #deterministic option for actions

N=1000#number of realizations

dt=1e-3

#q=1

steps=int(2e5) #walk lenght

params={'k':1,'eta':.1,'X_kunit':0.49,'theta':0.1}

#args={'feedback':feedback,'params':params}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,q=1e-4,dt=1e-3,plot=False,pow=0.5

#F=np.identity(2)

#env = make_vec_env(FisherEnv,n_envs=N,env_kwargs=args) 

nth=5
r_0=np.zeros(2)
der_0=np.zeros(2)
sc_0=(2*nth+1)*np.identity(2)
desc_0=np.zeros((2,2))

#bench_action=-0.1*np.ones((N,1))

data,data_bench=plt_2.main(N=N,timesteps=steps,agent=model,r_0=r_0,der_0=der_0,sc_0=sc_0,desc_0=desc_0,params=params,dt=dt)

#=pltools.allinone(env=env,timesteps=steps,agent=model,dumb_action=bench_action,noise=[N,0.3])
#data.to_csv(r'/home/fallani/prova/Fisher_nocost/DATA/agent_data_1000.csv', index = False)
#data_bench.to_json(r'/home/fallani/prova/Fisher_nocost/DATA/bench_data_1000.csv', index = False)

data.to_json('./DATA/data_agent.1_N{}_t{}.json'.format(N,steps))#,index=False)
data_bench.to_json('.DATA/data_bench.1_N{}_t{}.json'.format(N,steps))#,index=False)
#avg_ag.to_csv(r'/home/fallani/prova/Fisher_nocost/data/avg_agent.csv', index = False)
#avg_dumb.to_csv(r'/home/fallani/prova/Fisher_nocost/data/avg_dumb.csv', index = False)
#std_ag.to_csv(r'/home/fallani/prova/Fisher_nocost/data/std_agent.csv', index = False)
#std_dumb.to_csv(r'/home/fallani/prova/Fisher_nocost/data/std_dumb.csv', index = False)
end=time.time()
print(end-start)