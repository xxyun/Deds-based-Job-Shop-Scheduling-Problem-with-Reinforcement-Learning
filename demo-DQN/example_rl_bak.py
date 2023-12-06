import sys

import pandas as pd
from tool.env import JSP
from tool.fun import read_json
from tool.dqn import DQN
import time
import numpy as np
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")


def run_rl():
    config = {'device': 'cpu',
              'layer_sizes': (8, 32, 32, 32, 5),
              'memory_size': 20000,
              'batch_size': 1024,
              'epsilon': 0.5,
              'epsilon_decay_coefficient': 0.9,
              'epsilon_decay_interval': 100,
              'gamma': 0.9,
              'lr': 0.001,
              'l2': 0.0001,
              'target_replace_interval': 100}

    data = read_json("data/bak/machining_preparation_input.json")
    dqn = DQN(config)
    result = []
    reward_his = []
    ENV = JSP(data)
    start_time = time.time()
    for epoch in range(1):
        reward = 0
        actions = []
        env = deepcopy(ENV)
        # import streamlit as st
        # import datetime
        # from datetime import timedelta
        # import numpy as np
        # import plotly.express as px
        # fig = st.empty()
        # ST = datetime.datetime.now()
        dqn.add_episode()
        while True:
            s = env.state
            a = dqn.choose_action(s)
            s_, r, done = env.step(a)
            dqn.store_transition(s, a, r, s_, done)
            actions.append(a)
            reward += r
            if done:
                break
            # df = pd.DataFrame(env.logs, columns=["job_id", "机器名称", "st", "et"])
            # now = env.time_step * timedelta(days=1) + ST
            # df["st"] = df["st"] * timedelta(days=1) + ST
            # df["et"] = df["et"] * timedelta(days=1) + ST
            # df["color"] = np.where(df["et"] < now, '已完成', '进行中')
            # df["color"] = np.where(df["st"] > now, '已下发', df["color"])
            # df = df.sort_values(by=["机器名称", "color", "st"]).reset_index(drop=True)
            # plot = px.timeline(df, x_start="st", x_end="et", y="机器名称", color="color",
            #                    color_discrete_map={'已完成': '#A8AFB4', '进行中': '#42B8D1', '已下发': '#6BC273'},
            #                    width=1920 / 2, height=1080 / 2)
            # plot.add_vline(x=now, line_width=1, line=dict(color='red', width=1, dash='dash'), name='当前时刻')
            # plot.update_xaxes(
            #     tickformat="%m-%d %H"  # date format
            # )
            # fig.plotly_chart(plot)
        if dqn.memory_counter > dqn.batch_size:
            dqn.learn()
        end_time = time.time()
        reward_his.append(reward)
        result.append([epoch, dqn.epsilon, end_time, reward])
        print('epoch: %04d, epsilon:%.2f, reward:%.2f, reward_avg:%.2f' % (epoch, dqn.epsilon, reward, np.mean(reward_his[-50:])))
        pd.DataFrame(result, columns=["epoch", "epsilon", "time", "reward"]).to_csv("析取图.csv", index=False)


run_rl()
