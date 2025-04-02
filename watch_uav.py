######################################################################
# Verification of UAV track planning model based on DQN
# ---------------------------------------------------------------
# author by younghow
# email: younghowkg@gmail.com
# --------------------------------------------------------------
# 将训练好的DQN模型放入仿真模拟环境中进行测试,
# 可以使用env类中的reset_test函数对测试环境进行设置,
# 生成测试环境的UAV的起点与终点,并根据难度等级生成城市环境的建筑数目与风况.
# ----------------------------------------------------------------
# Put the trained DQN model into the simulation environment for testing.
# You can use the reset_test function in the env class to set the test environment,
# generate the starting point and end point of the UAV of the test environment,
# and generate the number of buildings,the number of buildings and wind conditions
#  in the urban environment according to the difficulty level.
##############################################################################
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import *
import torch
import os
import sys
LEARNING_RATE = 0.00033  # 学习率
num_episodes = 80000  # 训练周期长度
space_dim = 42  # n_spaces   状态空间维度
action_dim = 27  # n_actions   动作空间维度
threshold = 200
env = Env(space_dim, action_dim, LEARNING_RATE)

nogui = False
if len(sys.argv) == 2:
    nogui = True
    matplotlib.use("Agg")
    print("run in non-interactive mode")

def find_model_ckpt():
    # 查找 checkpoints 目录下的所有文件
    ckpt_path = "checkpoints"
    ckpt_list = os.listdir(ckpt_path)
    max_episode_ckpt = -1
    for ckpt in ckpt_list:
        if ckpt.endswith(".pth"):
            episode = ckpt.split(".")[0].split("_")[-1]
            if int(episode) > max_episode_ckpt:
                max_episode_ckpt = int(episode)

    qlocal_ckpt = "checkpoints/Qlocal_{}.pth".format(max_episode_ckpt)
    qtarget_ckpt = "checkpoints/Qtarget_{}.pth".format(max_episode_ckpt)
    return qlocal_ckpt, qtarget_ckpt


if __name__ == "__main__":

    qlocal_ckpt, qtarget_ckpt = find_model_ckpt()
    check_point_Qlocal = torch.load(qlocal_ckpt)
    check_point_Qtarget = torch.load(qtarget_ckpt)
    env.q_target.load_state_dict(check_point_Qtarget["model"])
    env.q_local.load_state_dict(check_point_Qlocal["model"])
    env.optim.load_state_dict(check_point_Qlocal["optimizer"])
    epoch = check_point_Qlocal["epoch"]
    # 真实场景运行
    env.level = 8  # 环境难度等级
    state = env.reset_test()  # 环境重置1
    total_reward = 0
    
    n_done = 0
    count = 0

    n_test = 1  # 测试次数
    n_creash = 0  # 坠毁数目
    frames = []  # 存储轨迹帧
    if not nogui:
        plt.ion()
    
    env.render(1, save_frames=True, frames=frames)
    print("正在运行模拟...")
    for i in range(n_test):
        while 1:
            if env.uavs[0].done:
                # 无人机已结束任务,跳过
                break
            action = env.get_action(
                torch.tensor(np.array([state[0]]), dtype=torch.float, device=device), 0.01
            )  # 根据Q值选取动作

            next_state, reward, uav_done, info = env.step(action.item(), 0)  # 根据选取的动作改变状态,获取收益

            total_reward += reward  # 求总收益
            # 交互显示
            # print(action)
            env.render(save_frames=True, frames=frames)
            # plt.pause(0.01)
            if uav_done:
                break
            if info == 1:
                success_count = success_count + 1

            state[0] = next_state  # 状态变更
        print(f"本次运行已结束, 无人机最终运行步数: {env.uavs[0].step}")
        print("正在保存并生成GIF...")
        env.ax.scatter(env.target[0].x, env.target[0].y, env.target[0].z, c="red")
        # plt.show()
        fig, ax = plt.subplots()
        ims = [[ax.imshow(frame, animated=True)] for frame in frames]  # 这里要用 `[]` 让 `ArtistAnimation` 解析
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        ani.save("result.gif", writer="pillow", dpi=300)
        print("本次运行结果已保存到 result.gif")

    plt.close('all')