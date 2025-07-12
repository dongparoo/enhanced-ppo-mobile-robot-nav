# replay_buffer.py
# -----------------------------------------------------------------------------
# Sum-Tree 기반 Prioritized Experience Replay (PER)
# -----------------------------------------------------------------------------
import numpy as np
from collections import namedtuple

Transition = namedtuple("Transition",
                        ["obs", "action", "logp",
                         "adv", "ret", "v_old"])

class SumTree:
    def __init__(self, capacity: int):
        # 다음 2의 거듭제곱보다 큰 전체 트리 크기
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity <<= 1
        self.tree = np.zeros(2 * self.capacity)
        self.data = [None] * self.capacity
        self.ptr = 0
        self.size = 0

    # -------------------------------------------------------------------------
    def _propagate(self, idx: int, change: float):
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # -------------------------------------------------------------------------
    def add(self, p: float, data: Transition):
        idx = self.ptr + self.capacity
        self.data[self.ptr] = data
        self.update(idx, p)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # -------------------------------------------------------------------------
    def _retrieve(self, idx: int, s: float):
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def get(self, s: float):
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    # -------------------------------------------------------------------------
    def sample(self, batch_size: int, beta: float):
        seg = self.total() / batch_size
        batch, idxs, is_weight = [], [], []
        priority_max = self.tree[1] if self.tree[1] != 0 else 1.0

        for i in range(batch_size):
            a, b = seg * i, seg * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.get(s)
            prob = p / priority_max
            w = (1.0 / (self.size * prob)) ** beta
            batch.append(data)
            idxs.append(idx)
            is_weight.append(w)

        is_weight = np.array(is_weight, dtype=np.float32)
        is_weight /= is_weight.max()  # normalise
        return idxs, batch, is_weight

# -----------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self,
                 capacity: int = 100_000,
                 alpha: float = 0.6,
                 beta_start: float = .4,
                 beta_frames: int = 50_000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.tree = SumTree(capacity)

    # -------------------------------------------------------------------------
    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start)
                   * self.frame / self.beta_frames)

    def add(self, td_error: float, transition: Transition):
        p = (abs(td_error) + 1e-5) ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size: int):
        beta = self._beta_by_frame()
        self.frame += 1
        idxs, batch, weights = self.tree.sample(batch_size, beta)
        return idxs, batch, weights, beta
    
        # -------------------------------------------------------------------------
    def prune(self, keep_ratio: float = 0.5):
        """TD-error 기준 상위 keep_ratio 만큼만 유지."""
        if self.tree.size == 0:
            return
        priorities = self.tree.tree[
            self.tree.capacity : self.tree.capacity + self.tree.size
        ]
        cutoff = np.quantile(priorities, 1 - keep_ratio)
        new_transitions = [
            self.tree.data[i] for i, p in enumerate(priorities) if p >= cutoff
        ]
        cap = self.tree.capacity
        self.__init__(capacity=cap,
                    alpha=self.alpha,
                    beta_start=self.beta_start,
                    beta_frames=self.beta_frames)
        for tr in new_transitions:
            self.add(1.0, tr)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = (abs(err) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
