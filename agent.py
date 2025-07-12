import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PPO.model          import ActorCriticNet
from PPO.replay_buffer  import PrioritizedReplayBuffer, Transition

class PPOAgent:
    """
    풀-스택(PER - Dynamic βt - CVaR - Potential Shaping) PPO 에이전트.
    * on-policy → PER 편향을 IS-weight 로완화
    * βt(t, d, ρ)  : 동적 엔트로피 계수
    * CVaR penalty : 낮은 리턴에 위험 패널티
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 lidar_dim: int = 30,
                 buffer_size: int = 100_000,
                 batch_size: int = 512,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.15):

        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps

        # 네트워크 & 옵티마이저 ------------------------------------------------
        self.net       = ActorCriticNet(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

        # PER -----------------------------------------------------------------
        self.buffer      = PrioritizedReplayBuffer(capacity=buffer_size)
        self.batch_size  = batch_size

        # Dynamic-Entropy βt ---------------------------------------------------
        self.beta_max, self.beta_min = 0.20, 0.03
        self.lambda_d, self.lambda_rho = 0.25, 0.2
        self.cur_step    = 0
        self.beta_t      = self.beta_max

        # Risk(CVaR) ----------------------------------------------------------
        self.lambda_risk = 0.5
        self.cvar_alpha  = 0.1                  # 하위 10 % 구간

        self.max_updates  = 490   # ← 하이퍼파라미터. 실험 길이에 맞춰 조정
        self.update_count = 0

        self.lidar_dim = lidar_dim

    # -------------------------------------------------------------------------
    # 모델·상태 저장/로드
    # -------------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "net":            self.net.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "cur_step":       self.cur_step,
            "update_count":   self.update_count,
            "beta_t":         self.beta_t,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.cur_step     = ckpt.get("cur_step",     0)
        self.update_count = ckpt.get("update_count", 0)
        self.beta_t       = ckpt.get("beta_t",       self.beta_max)

    # -------------------------------------------------------------------------
    # βt(t, d, ρ) 계산
    # -------------------------------------------------------------------------
    def _update_beta(self, t_ratio: float, d_t: float, rho_t: float) -> float:
        beta     = self.beta_max + (self.beta_min - self.beta_max) * (t_ratio ** 2)
        beta    += self.lambda_d * d_t - self.lambda_rho * rho_t
        self.beta_t = float(np.clip(beta, self.beta_min, self.beta_max))
        return self.beta_t

    # -------------------------------------------------------------------------
    # 버퍼 I/O
    # -------------------------------------------------------------------------
    def store(self,
              obs, action, logp,
              adv, ret, v_old,
              td_error):
        self.buffer.add(td_error,
                        Transition(obs, action, logp,
                                   adv, ret, v_old))

    def _prepare_minibatch(self):
        idxs, batch, weights, _ = self.buffer.sample(self.batch_size)
        extract = lambda field: np.stack([getattr(b, field) for b in batch])

        obs   = torch.as_tensor(extract("obs"),   dtype=torch.float32)
        act   = torch.as_tensor(extract("action"),dtype=torch.float32)
        logp  = torch.as_tensor(extract("logp"),  dtype=torch.float32)
        adv   = torch.as_tensor(extract("adv"),   dtype=torch.float32)
        ret   = torch.as_tensor(extract("ret"),   dtype=torch.float32)
        v_old = torch.as_tensor(extract("v_old"), dtype=torch.float32)
        w     = torch.as_tensor(weights,          dtype=torch.float32)

        return idxs, obs, act, logp, adv, ret, v_old, w

    # -------------------------------------------------------------------------
    # 학습 업데이트
    # -------------------------------------------------------------------------
    def update(self):
        self.update_count += 1
        t_ratio = min(1.0, self.update_count / self.max_updates)

        if self.buffer.tree.size < self.batch_size:
            return {}

        idxs, obs, actions, logp_old, adv, returns, v_old, weights = \
            self._prepare_minibatch()

        self.cur_step += len(obs)

        # Dynamic βt 계산
        lidar = obs[:, :self.lidar_dim]
        rho_t = (lidar < 1.0).float().mean().item()
        d_t   = obs[:, -2].mean().item() / 10.0
        beta  = self._update_beta(t_ratio, d_t, rho_t)

        # Advantage 정규화
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # Policy / Value / Entropy
        logp, entropy, value_pred = self.net.evaluate(obs, actions)
        ratio      = torch.exp(logp - logp_old)
        surr1      = ratio * adv
        surr2      = torch.clamp(ratio,
                                 1.0 - self.clip_eps,
                                 1.0 + self.clip_eps) * adv
        policy_loss = -((torch.min(surr1, surr2) + beta * entropy)
                        * weights).mean()
        value_loss  = ((returns - value_pred) ** 2 * weights).mean()

        # CVaR Risk Loss
        threshold = torch.quantile(returns, self.cvar_alpha)
        risk_term = torch.relu(threshold - returns)
        risk_loss = self.lambda_risk * (risk_term * weights).mean()

        total_loss = policy_loss + value_loss + risk_loss

        # Back-prop
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.3)
        self.optimizer.step()

        if self.update_count % 2000 == 0:
            self.beta_t = 0.15

        # PER priority 업데이트
        with torch.no_grad():
            _, _, new_val = self.net.evaluate(obs, actions)
        td_new = returns - new_val
        self.buffer.update_priorities(idxs, td_new.detach().cpu().numpy())

        return {
            "total":  total_loss.item(),
            "policy": policy_loss.item(),
            "value":  value_loss.item(),
            "risk":   risk_loss.item(),
            "beta":   beta,
            "rho":    rho_t,
            "d":      d_t
        }
