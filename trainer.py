# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:33:07 2020

@author: lvjf

train the agent
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from job_env import job_shop_env
from RL_brain import ActorCritic
from utils import v_wrap
import csv

def train(args):
    torch.manual_seed(args.seed)

    env = job_shop_env()
    
    model = ActorCritic(env.state_dim, env.action_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = v_wrap(state)
    done = True #当前回合（episode）是否结束
    action_dim = env.expert #行动维度，或者理解成可以行动的可选项（133个专家可选）

    episode_length = 0 #计数器，用于跟踪当前回合的长度或所执行的步数。
    complete_jobs = [] #完成的作业
    expert_complete_job = [] #完成作业的专家
    complete_job_start_time = []  #作业开始的时间
    update_list = [] #已完成的作业列表，用于更新环境或记录完成的作业
    # 初始化控制
    prev_job_end_time = None
    current_job_end_time = None
    
    for episode in range(args.episode):
        
        #长短期记忆网络（LSTMs），代表隐藏状态（hx）和单元状态（cx）
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        
        if len(complete_jobs) != 0:
            update_list = [n for m in complete_jobs for n in m]
            env.update(update_list)

        values = []
        log_probs = []
        rewards = []
        entropies = []
        #初始化original_job_end_time
        original_job_end_time = []
        


        for step in range(args.num_steps+1):
            episode_length += 1
            print('\n\ncurrent_step', step)
            # print('current:',current_job_end_time)
            # print('prev:',prev_job_end_time)
        


            
            # action：由模型选择的动作。
            # log_prob：所选择动作的对数概率，用于策略梯度计算。
            # entropy：动作概率分布的熵，用于鼓励探索。
            # value：评论家网络估计的状态值，用于计算优势函数和值损失。
            # print('now:',state)
            action, log_prob, entropy, value = model.choose_action((state, (hx,cx)),action_dim)
            
            if prev_job_end_time is not None and current_job_end_time == prev_job_end_time:
                # 如果 job_start_time 没有更新，则将 action 设置为 0
                action = torch.zeros_like(action)
                print('no pass')
            else:
                print('pass!')
                

            log_prob = log_prob.gather(1, action)[0]

            # print('action:',action)
            # print('log_prob:',log_prob)
            # print('entropy:',entropy)
            # print('value:',value)

            # done_job：完成的作业列表。
            # done_expert：完成相应作业的专家列表。
            # job_start_time：每项作业开始处理的时间。

            
            # import pdb; pdb.set_trace()
            state, reward, done, done_job, done_expert, job_start_time, job_end_time = env.step(action.view(-1,).numpy(),step)
            print('job_start_time:',job_start_time)
            # print('before_job_end_time:', job_end_time)
            # print('before_current_job_end_time:', current_job_end_time)
            # print('prev_job_end_time:', prev_job_end_time)
            
            if job_end_time != prev_job_end_time:
                # 只有在 job_end_time 发生变化时才更新 prev_job_end_time
                prev_job_end_time = current_job_end_time
                current_job_end_time = job_end_time.copy()
                
                print('到达决策步！')
                print('job_end_time:',job_end_time)
                print('current_job_end_time:',current_job_end_time)
                print('prev_job_end_time:',prev_job_end_time)
            
            
            # if job_end_time != original_job_end_time:
            #     #判断列表是否更新，更新则重新预测一遍装备
            #     print('到达决策步！')
            #     print('original_job_end_time:',original_job_end_time)
            #     print('job_end_time:',job_end_time)
            #     # print('after:',torch.tensor(state).float())
            #     action, log_prob, entropy, value = model.choose_action((torch.tensor(state).float(), (hx,cx)),action_dim)
            #     log_prob = log_prob.gather(1, action)[0]
            #     state, reward, done, done_job, done_expert, job_start_time, job_end_time = env.step(action.view(-1,).numpy())
            #     # 记录更新的 job_end_time 列表
            #     original_job_end_time = job_end_time.copy()
            

            done = done or episode_length >= args.max_episode_length
            ## reward shaping 确保奖励值在 -1 到 1 之间。奖励整形可以帮助提高训练的稳定性和效率。
            reward = max(min(reward, 1), -1)
            if episode_length % 20 == 0:
                print('reward=',reward)
                #print(done_job)
            if done:
                complete_jobs.append(done_job)
                expert_complete_job.append(done_expert)
                complete_job_start_time.append(job_start_time)
                print('Complete these jobs with 100 iterations:')
                print(complete_jobs)
                print('Current episode:',episode)
                episode_length = 0
                state = env.reset()

            state = v_wrap(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            if done:
                break
        
        if len(list(set(update_list))) > 8800:
            ## write results into the csv file
            with open('submit_{}.csv'.format(len(list(set(update_list)))),'w') as f:
                writer = csv.writer(f)
                for i in range(len(complete_jobs)):
                    for j in range(len(complete_jobs[i])):
                        writer.writerow([complete_jobs[i][j]+1, expert_complete_job[i][j]+1, complete_job_start_time[i][j]])

        if episode == args.episode -1 or len(list(set(update_list))) == 8840:
            ## write results into the csv file
            with open('submit.csv','w') as f:
                writer = csv.writer(f)
                for i in range(len(complete_jobs)):
                    for j in range(len(complete_jobs[i])):
                        writer.writerow([complete_jobs[i][j]+1, expert_complete_job[i][j]+1, complete_job_start_time[i][j]])
            break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward(torch.ones_like(policy_loss))
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        print(policy_loss.mean() + args.value_loss_coef * value_loss)
        print('para updated')
