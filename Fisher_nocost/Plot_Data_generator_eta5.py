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

plt.style.use(['seaborn-whitegrid'])#(['Solarize_Light2'])

det=False #deterministic option for actions

N=10000#number of realizations

dt=1e-3

steps=int(2e5) #walk lenght



nth=5
r_0=np.zeros(2)
der_0=np.zeros(2)
sc_0=(2*nth+1)*np.identity(2)
desc_0=np.zeros((2,2))

params={'k':1,'eta':.5,'X_kunit':0.49,'theta':0.1}

model=PPO2.load('/home/fallani/prova/Fisher_nocost/Fisher_checkpoint/Fisher_tests_BayesRK4_ci_physics/feedBayes_steadyTrue_lro0.00025_ts30.0M_N4_ec0.001_0.45_1e5_theta0.1_eta.5_Custom6464_1e-3_RK4_nonorm_full_inf_s2_q0/rl_model_29900000_steps.zip')

data,data_bench1,data_bench2,data_bench3=plt_2.main_trajs(N=N,timesteps=steps,agent=model,markevery=1000,r_0=r_0,der_0=der_0,sc_0=sc_0,desc_0=desc_0,params=params,dt=dt,bench=-0.1,bench_noise=0.1)


data.to_json('/home/fallani/prova/Fisher_nocost/dati2/data_meanbootm1000_agent5_N{}_t{}.json'.format(N,steps))#,index=False)
data_bench1.to_json('/home/fallani/prova/Fisher_nocost/dati2/data_meanbootm1000_benchnoise_5_N{}_t{}.json'.format(N,steps))#,index=False)
data_bench2.to_json('/home/fallani/prova/Fisher_nocost/dati2/data_meanbootm1000_benchdumb_5_N{}_t{}.json'.format(N,steps))#,index=False)
data_bench3.to_json('/home/fallani/prova/Fisher_nocost/dati2/data_meanbootm1000_benchnull_5_N{}_t{}.json'.format(N,steps))#,index=False)
