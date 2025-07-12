#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# 학습 재개 스크립트 (지정한 체크포인트 사용)
# -----------------------------------------------------------------------------
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import rclpy
import os

from omni.isaac.kit import SimulationApp
from PPO.goal_nav_env import GoalNavEnv
from PPO.agent        import PPOAgent
from PPO.utils        import compute_gae

def main():
    # ─── Isaac Sim headless App 생성 (physics-only stepping) ───────────────
    sim_app = SimulationApp({
        "headless": True,
        "timeScale": 5.0,
        "exts": ["omni.isaac.ros_bridge"],
    })

    rclpy.init(args=None)

    # ─── 환경 & 에이전트 초기화 ───────────────────────────────────────────
    env     = GoalNavEnv(expand_episodes=1200)
    env.curr_expand_step = 618
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent   = PPOAgent(obs_dim, act_dim, lidar_dim=env.lidar_dim)
    logger  = SummaryWriter("runs/per_risk_beta_full")

    # ─── (추가) 재개할 체크포인트 & 시작 스텝/에피소드 지정 ─────────────────   
    resume_ckpt       = "checkpoints/ppo_step_3020000.pt"
    start_global_step = 3020000
    start_episode     = 8787

    global_step = 0
    episode     = 0
    if os.path.isfile(resume_ckpt):
        print(f"[Resume] Loading checkpoint → {resume_ckpt}")
        agent.load(resume_ckpt)
        global_step = start_global_step
        episode     = start_episode
        env.episode_count = episode
    else:
        print(f"[Resume] 체크포인트를 찾을 수 없어, 처음부터 시작합니다.")

    # ─── 체크포인트 저장 설정 ───────────────────────────────────────────────
    save_interval = 5000
    save_dir      = os.path.dirname(resume_ckpt) or "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # ─── 학습 변수 ────────────────────────────────────────────────────────
    max_steps = 5_000_000
    max_episodes  = 10_000
    obs       = env.reset()
    ep_ret    = 0.0
    ep_step   = 0

    traj_obs, traj_act, traj_logp = [], [], []
    traj_rew, traj_done, traj_val   = [], [], []

    # ─── 메인 학습 루프 ──────────────────────────────────────────────────
    while sim_app.is_running() and global_step < max_steps:
        # 액션 선택
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, logp, _ = agent.net.act(obs_t)

        # 물리 fast-forward
        sim_app.update()

        # 환경 스텝
        next_obs, reward, done, info = env.step(action.squeeze(0).numpy())
        with torch.no_grad():
            next_val = agent.net.critic(
                torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            ).item()
        value_now = agent.net.critic(obs_t).item()

        # 에피소드 스텝 증가
        ep_step += 1

        # rollout 임시 버퍼에 저장
        traj_obs.append(obs)
        traj_act.append(action.squeeze(0).numpy())
        traj_logp.append(logp.item())
        traj_rew.append(reward)
        traj_done.append(done)
        traj_val.append(value_now)


        # 로그 출력
        robot_x, robot_y = env.pose[:2]
        goal_x, goal_y   = env.goal
        dist             = np.linalg.norm(env.goal - env.pose[:2])
        min_lidar        = float(np.min(env.scan))
        print(f"{global_step:7d} | Ep{episode:4d} | step={ep_step:3d} | "
              f"expand={env.curr_expand_step:4d} span={env.prev_span:5.2f} | "
              f"XY=({robot_x:5.2f},{robot_y:5.2f}) | "
              f"Goal=({goal_x:5.2f},{goal_y:5.2f}) | "
              f"Dist={dist:5.2f}  LidMin={min_lidar:4.2f}  R={reward:5.2f}")

        # 배치가 가득 찼으면 업데이트
        if len(traj_obs) >= agent.batch_size:
            metrics = agent.update()
            if metrics:
                for k, v in metrics.items():
                    logger.add_scalar(f"loss/{k}", v, global_step)

        # 다음 스텝 준비
        obs         = next_obs
        ep_ret     += reward
        global_step += 1

        # 체크포인트 저장 (5000스텝마다)
        if global_step % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"ppo_step_{global_step}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] step {global_step} 모델 저장: {ckpt_path}")

        # 에피소드 종료 처리
        if done:
            last_val = next_val if not info.get("timeout", False) else 0.0
            adv, ret = compute_gae(traj_rew, traj_val, traj_done,
                                   last_val, agent.gamma, agent.lam)
            for i in range(len(traj_obs)):
                td_err = ret[i] - traj_val[i]
                agent.store(traj_obs[i], traj_act[i], traj_logp[i],
                            adv[i].item(), ret[i].item(),
                            traj_val[i], td_err.item())

            # 임시 버퍼 초기화
            traj_obs.clear(); traj_act.clear(); traj_logp.clear()
            traj_rew.clear(); traj_done.clear(); traj_val.clear()

            # 에피소드 로그
            logger.add_scalar("episode/return",  ep_ret, episode)
            logger.add_scalar("episode/steps",   ep_step, episode)
            logger.add_scalar("episode/success", int(info.get("success", False)), episode)
            print(f"[Episode {episode:4d}] steps={ep_step:3d} return={ep_ret:7.2f} "
                  f"{'SUCCESS' if info.get('success', False) else 'FAIL'}")

            # 에피소드 카운트 및 리셋
            episode += 1
            env.episode_count = episode

            # 10 000 에피소드 달성했으면 루프 탈출
            if episode >= max_episodes:  
                print(f"[Finish] {episode} episodes reached — training stopped.")
                break

            obs, ep_ret, ep_step = env.reset(), 0.0, 0

    # 종료
    logger.close()
    rclpy.shutdown()
    sim_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user — shutting down.")
        rclpy.shutdown()
