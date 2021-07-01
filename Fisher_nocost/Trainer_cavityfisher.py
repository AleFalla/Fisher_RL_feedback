import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1,PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.common import make_vec_env
from fisher_env import FisherEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

#define custom network
b=64
#class CustomPolicy(FeedForwardPolicy):
#    def __init__(self, *args, **kwargs):
#        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[dict(pi=[b,b],
#                                                          vf=[b,b])],feature_extraction="mlp")
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,net_arch=[512,dict(pi=[256,128],
                                                          vf=[256,128])],feature_extraction="mlp")

e_c=0.0001 #define entropy coeff
feedback='Bayes' #'Markov' or 'Bayes'
steady=True #if True resets always with steady state conditions
N=8 #number of parallel workers
LRo=2.5e-4  #learning rate                        
#uact=True #if we want to use u as action (only Bayesian)
TIMESTEPS=int(50e6) #training steps
sched_LR=LinearSchedule(1,LRo,0) #lr schedule
LR=sched_LR.value 
qs=0 #no feedback cost 
dirname='Fisher_tests_{}RK4_cirand'.format(feedback) #directory name
title='feed{}_steady{}_lro{}_ts{}M_N{}_ec{}_0.49_3e4_theta0.1_Mlp_1e-3_RK4_bothrand_s2'.format(feedback,steady,LRo,TIMESTEPS/1e6,N,e_c)
#make checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=int(100000/N), save_path='./Fisher_nocost_checkpoint/{}/{}_q{}'.format(dirname,title,qs))
callback = checkpoint_callback
#set parameters and start training
params={'k':1,'eta':1,'X_kunit':0.49,'theta':0.1} #if a parameter is set to None it will be sampled from a uniform distribution at every reset
args={'feedback':feedback,'params':params}#i parametri di default son questi: rewfunc=Tools.purity_like_rew,q=1e-4,dt=1e-3,plot=False,pow=0.5
#instantiate environment
env = make_vec_env(FisherEnv,n_envs=N,env_kwargs=args) 
#instantiate model
model=PPO2(MlpPolicy,env,n_steps=128,learning_rate=LR,lam=0.95,ent_coef=e_c,verbose=1,nminibatches=4,noptepochs=4,tensorboard_log='./Fisher_nocost_TRAIN_LOG/{}/{}_q{}'.format(dirname,title,qs),seed=2)
#train the model
model.learn(total_timesteps=TIMESTEPS,callback=callback,tb_log_name='{}_q{}'.format(title,qs))
#save the trained model at a given path
model.save('./MODELS/{}/{}_q{}'.format(dirname,title,qs))
        
