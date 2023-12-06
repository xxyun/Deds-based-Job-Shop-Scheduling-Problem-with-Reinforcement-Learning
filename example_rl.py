import pandas as pd
# from tool.fun import read_json
from job_env import job_shop_env
from RL_brain import ActorCritic
from utils import v_wrap
import csv
import time
import numpy as np
import warnings
from copy import deepcopy
import torch.optim as optim
import torch
warnings.filterwarnings("ignore")


def run_rl():
    config = {'num-steps': 100,
              'max-episode-length': 100000,
              'gamma': 0.9,
              'lr': 0.01,
              'seed': 2020}

    torch.manual_seed(config['seed'])
    result = []
    reward_his = []
    ENV = job_shop_env()
    dqn = ActorCritic(ENV.state_dim, ENV.action_dim)
    start_time = time.time()
    optimizer = optim.Adam(dqn.parameters(), lr=config['lr'])
    
    dqn.train()
    state = ENV.reset()
    state = v_wrap(state)
    done = True #当前回合（episode）是否结束
    action_dim = ENV.expert #行动维度，或者理解成可以行动的可选项（133个专家可选）
    episode_length = 0 #计数器，用于跟踪当前回合的长度或所执行的步数。
    complete_jobs = [] #完成的作业
    expert_complete_job = [] #完成作业的专家
    complete_job_start_time = []  #作业开始的时间
    update_list = [] #已完成的作业列表，用于更新环境或记录完成的作业
    # 初始化控制
    prev_job_end_time = None
    current_job_end_time = None
    
    for epoch in range(5000):
        reward = 0
        actions = []
        env = deepcopy(ENV)
        import streamlit as st
        import datetime
        from datetime import timedelta
        import numpy as np
        import plotly.express as px
        fig = st.empty()
        ST = datetime.datetime.now()
        # dqn.add_episode()
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
        while True:
            # s = env.state
            # a = dqn.choose_action(s)
            # s_, r, done = env.step(a)
            # dqn.store_transition(s, a, r, s_, done)
            # actions.append(a)
            # reward += r
            # if done:
            #     break
            
        # for episode in range(args.episode):
            
            #长短期记忆网络（LSTMs），代表隐藏状态（hx）和单元状态（cx）

            for step in range(config['num-steps']+1):
                episode_length += 1
                print('\n\ncurrent_step', step)
            
                action, log_prob, entropy, value = dqn.choose_action((state, (hx,cx)),action_dim)
                
                if prev_job_end_time is not None and current_job_end_time == prev_job_end_time:
                    # 如果 job_start_time 没有更新，则将 action 设置为 0
                    action = torch.zeros_like(action)
                    print('no pass')
                else:
                    print('pass!')
                    
                log_prob = log_prob.gather(1, action)[0]
                state, reward, done, done_job, done_expert, job_start_time, job_end_time = env.step(action.view(-1,).numpy(), step)
                print('job_start_time:',job_start_time)

                if job_end_time != prev_job_end_time:
                    # 只有在 job_end_time 发生变化时才更新 prev_job_end_time
                    prev_job_end_time = current_job_end_time
                    current_job_end_time = job_end_time.copy()
                    
                    print('到达决策步！')
                    print('job_end_time:',job_end_time)
                    print('current_job_end_time:',current_job_end_time)
                    print('prev_job_end_time:',prev_job_end_time)
                
        

                done = done or episode_length >= config['max-episode-length']
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
                    print('Current epoch:',epoch)
                    episode_length = 0
                    state = env.reset()

                state = v_wrap(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                if done:
                    break

                # df = pd.DataFrame([[0, 0, 0, 0]], columns=["job_id", "机器名称", "st", "et"])
                # df = df.append({"job_id": 0, "机器名称": 0, "st": 0, "et": 0}, ignore_index=True)
                # 创建一个空的 DataFrame，列名与 env.logs 相对应
                df = pd.DataFrame(columns=["job_id", "机器名称", "st", "et"])
                # 检查 env.logs 是否为空
                if not env.logs:
                    # env.logs 为空，添加一行默认值
                    df = df._append({"job_id": 0, "机器名称": '0', "st": 0, "et": 0}, ignore_index=True)
                else:
                    # 遍历 env.logs 中的每条记录
                    for log in env.logs:
                        # 解包 log 中的内容，如果属性为空，则使用 0
                        job_id, machine_name, st, et = log
                        job_id = 0 if not job_id.all() else job_id
                        machine_name = '0' if not machine_name  else machine_name
                        st = 0 if not st  else st
                        et = 0 if not et  else et

                        # 将数据追加到 DataFrame 中
                        # df = df._append({"job_id": job_id, "机器名称": machine_name, "st": st, "et": et}, ignore_index=True)       
                        df = df._append({"job_id": job_id, "机器名称": 'A%s' % machine_name, "st": st, "et": et}, ignore_index=True)          
                
                now = step * timedelta(seconds=1) + ST
                # import pdb; pdb.set_trace()  
                # 转换数据类型之前，确保所有非数值数据都被处理
                df["st"] = pd.to_numeric(df["st"], errors='coerce').fillna(0)
                df["et"] = pd.to_numeric(df["et"], errors='coerce').fillna(0)
                df['job_id'] = df['job_id'].astype('string')
                df['机器名称'] = df['机器名称'].astype('string')
                #现在进行转换
                df["st"] = df["st"] * timedelta(seconds=1) + ST
                df["et"] = df["et"] * timedelta(seconds=1) + ST
                df["color"] = np.where(df["et"] < now, '已完成', '进行中')
                df["color"] = np.where(df["st"] > now, '已下发', df["color"])
                # 将 datetime 转换为字符串
                # df["st"] = df["st"].dt.strftime('%Y-%m-%d %H:%M:%S')
                # df["et"] = df["et"].dt.strftime('%Y-%m-%d %H:%M:%S')
                # df["st"] = pd.to_numeric(df["st"], errors='coerce').fillna(0)
                # df["et"] = pd.to_numeric(df["et"], errors='coerce').fillna(0)
                
                # df["st"] = pd.to_datetime(df["st"], errors='coerce')
                # df["et"] = pd.to_datetime(df["et"], errors='coerce')
                # import pdb; pdb.set_trace()
                df = df.sort_values(by=["机器名称", "color", "st"]).reset_index(drop=True)
                
                plot = px.timeline(df, x_start="st", x_end="et", y="机器名称", color="color",
                                color_discrete_map={'已完成': '#A8AFB4', '进行中': '#42B8D1', '已下发': '#6BC273'},
                                width=1920 / 2, height=1080 / 2)
                plot.add_vline(x=now, line_width=1, line=dict(color='red', width=1, dash='dash'), name='当前时刻')
                plot.update_xaxes(
                    tickformat="%m-%d %H"  # date format
                )
                # import pdb; pdb.set_trace()
                fig.plotly_chart(plot)
                
            # if dqn.memory_counter > dqn.batch_size:
            #     dqn.train()
            # end_time = time.time()
            # reward_his.append(reward)
            # result.append([epoch, end_time, reward])
            # print('epoch: %04d, reward:%.2f, reward_avg:%.2f' % (epoch, reward, np.mean(reward_his[-50:])))
            # pd.DataFrame(result, columns=["epoch", "time", "reward"]).to_csv("离散仿真test.csv", index=False)


run_rl()
