#!/usr/bin/env python3
# -----------------------------------------------------------------------------

# 학습 루프 (풀-스택)   2025-06-06 full replacement (fixed)
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
        "timeScale": 5.0,    # 시뮬레이션을 5배 빠르게
        })


    rclpy.init(args=None)

    # ─── 환경 & 에이전트 초기화 ───────────────────────────────────────────
    env     = GoalNavEnv(expand_episodes=9600)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent   = PPOAgent(obs_dim, act_dim, lidar_dim=env.lidar_dim)
    logger  = SummaryWriter("runs/per_risk_beta_full")

    # ─── 체크포인트 저장 설정 ───────────────────────────────────────────────
    save_interval = 5000                       # 몇 스텝마다 저장할지
    save_dir      = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # ─── 학습 변수 ────────────────────────────────────────────────────────
    global_step = 0
    episode     = 0
    max_steps   = 1_000_000                    # 총 환경 스텝

    obs      = env.reset()
    ep_ret   = 0.0
    ep_step  = 0

    # rollout 임시 버퍼
    traj_obs, traj_act, traj_logp = [], [], []
    traj_rew, traj_done, traj_val   = [], [], []


    warmup_epochs = 5

    # ─── 메인 학습 루프 ──────────────────────────────────────────────────
    while sim_app.is_running() and global_step < max_steps:
        # 액션 선택
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, logp, _ = agent.net.act(obs_t)

        # ─── 물리 스텝만 fast-forward ─────────────────────────────────────
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

        # rollout 버퍼에 저장
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
              f"XY=({robot_x:5.2f},{robot_y:5.2f}) "
              f"Goal=({goal_x:5.2f},{goal_y:5.2f}) "
              f"Dist={dist:5.2f}  LidMin={min_lidar:4.2f}  R={reward:5.2f}")

        # 배치가 가득 찼으면 업데이트
        if len(traj_obs) >= agent.batch_size:
            # 버퍼 워밍업 체크
            if agent.buffer.tree.size >= agent.batch_size * warmup_epochs:
                metrics = agent.update()
                if metrics:
                    for k, v in metrics.items():
                        logger.add_scalar(f"loss/{k}", v, global_step)
                if agent.update_count % 500 == 0:
                    agent.buffer.prune(keep_ratio=0.5)
            else:
                # 워밍업 중이라면 스킵 (원하면 로그만 남기기)
                if global_step % 1000 == 0:
                    print(f"[워밍업 중] buffer size: {agent.buffer.tree.size}")            

        # 다음 스텝 준비
        obs         = next_obs
        ep_ret     += reward
        global_step += 1

        # ─── 체크포인트 저장 ────────────────────────────────────────────
        if global_step % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f"ppo_step_{global_step}.pt")
            agent.save(ckpt_path)
            print(f"[Checkpoint] step {global_step} 모델 저장: {ckpt_path}")

        # ─── 에피소드 종료 처리 ───────────────────────────────────────────
        if done:
            # GAE 계산 후 PER에 저장
            last_val = next_val if not info.get("timeout", False) else 0.0
            adv, ret = compute_gae(traj_rew, traj_val, traj_done,
                                   last_val, agent.gamma, agent.lam)
            for i in range(len(traj_obs)):
                td_err = ret[i] - traj_val[i]
                agent.store(traj_obs[i], traj_act[i], traj_logp[i],
                            adv[i].item(), ret[i].item(),
                            traj_val[i], td_err.item())

            # 버퍼 초기화
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
