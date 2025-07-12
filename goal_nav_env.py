# goal_nav_env.py  ★2025-06-09 LiDAR30 & Accel-Action
import rclpy, math, time, gym, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg   import Odometry
from geometry_msgs.msg import Twist, Pose2D
from threading import Thread, Event
from collections import deque

MAP_ORIGIN = np.array([-9.0, -9.0], dtype=np.float32)

class GoalNavEnv(Node, gym.Env):
    """Isaac Sim ↔ ROS2 Scout Mini 용 PPO 학습 환경 (LiDAR 30 + Accel Action)."""
    # ------------------------------ 초기화 -----------------------------------
    def __init__(self, expand_episodes: int = 9600):
        Node.__init__(self, "goal_nav_env")
        gym.Env.__init__(self)

        # ── ROS I/F ─────────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry,  '/odom', self.odom_cb, 10)
        self.cmd_pub          = self.create_publisher(Twist,  '/cmd_vel',          10)
        self.reset_robot_pub  = self.create_publisher(Pose2D, '/reset_robot_pose', 10)
        self.reset_goal_pub   = self.create_publisher(Pose2D, '/reset_goal_pose',  10)

        # ── 파라미터 ───────────────────────────────────────────────────────
        self.lidar_dim = 30          # ↓ 180 → 30
        self.fov_deg   = 360         # 전방 180°만 사용 (원하면 120 등으로 축소)
        self.max_range = 10.0
        self.dt        = 0.1
        self.step_limit = None
        self.gamma     = 0.99
        self.expand_eps = expand_episodes
        self.start_span = 6.0    # 에피소드 0에서 span = 6.0  → x∈[-9, -3]
        self.max_span   = 18.0   # 최종적으로 span = 18.0 → x∈[-9, +9]

        self.curr_expand_step = 0            # 실제로 늘린 스텝 카운터
        self.freeze_span      = False          # ──★ 새 플래그
        self.success_window   = deque(maxlen=200)
        self.success_buffer   = deque(maxlen=10)  # 최근 30 에피소드 성공 기록
        self.success_thresh   = 0.7          # 80% 이상일 때 난이도 UP

        # 속도 한계
        self.v_max, self.w_max = 1.5, 4.0 
        # 현재 명령 속도
        self.v, self.w = 0.0, 0.0 

        # ── 내부 상태 ─────────────────────────────────────────────────────
        self.scan  = np.full(self.lidar_dim, self.max_range, np.float32)
        self.pose  = np.zeros(3, np.float32)      # (x, y, yaw)
        self.goal  = np.zeros(2, np.float32)      # 목표 odom 좌표
        self.state_ev = Event()
        self.episode_count = 0
        Thread(target=rclpy.spin, args=(self,), daemon=True).start()

        # ── Gym Space ──────────────────────────────────────────────────────
        obs_dim   = self.lidar_dim + 6            # LiDAR + v ω + d θ
        act_high  = np.array([self.v_max, self.w_max], np.float32)  # v,w
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-0.2, -self.w_max], np.float32),  # 후진 금지
            high=act_high, dtype=np.float32)

        # ── 보상 변수 ──────────────────────────────────────────────────────
        self.prev_potential = 0.0
        self.prev_goal_dist = 0.0
        self.lock_steps     = 0 
        self.steps          = 0

    # ---------------------------- ROS Callbacks ------------------------------
    def scan_cb(self, msg: LaserScan):
        raw = np.asarray(msg.ranges, dtype=np.float32)
        raw[np.isinf(raw)] = self.max_range

        # ① FOV 자르기 (전방만)
        if self.fov_deg < 360:
            half = int(len(raw) * self.fov_deg / 360 / 2)
            center = len(raw) // 2
            raw = raw[center-half:center+half]

        # ② 균등 다운샘플링 → lidar_dim
        idx = np.linspace(0, raw.size - 1, self.lidar_dim).astype(np.int32)
        self.scan = np.clip(raw[idx], 0.0, self.max_range)
        self.state_ev.set()

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                         1 - 2*(q.y*q.y + q.z*q.z))
        self.pose = np.array([p.x, p.y, yaw], np.float32)

    # ------------------------- Gym helpers -----------------------------------
    def get_obs(self):
        # 목표까지 상대 벡터
        goal_vec = self.goal - self.pose[:2]
        dist = np.linalg.norm(goal_vec)
        dist_norm  = dist / (self.init_dist + 1e-6)
        goal_norm  = goal_vec / (self.init_dist + 1e-6)
        theta = math.atan2(goal_vec[1], goal_vec[0]) - self.pose[2]
        theta = math.atan2(math.sin(theta), math.cos(theta))   # [-π, π]

        return np.concatenate([self.scan,
                            [self.v, self.w],
                            [dist_norm, theta],
                            goal_norm]).astype(np.float32)
        

    # --------------------------- Reward & Done -------------------------------
    def compute_reward(self, obs):
        """
        보상 = 진행(progress) + 헤딩 정렬 + 부드러운 가속(패널티) +
            충돌 위험(패널티) + 시간 패널티
        모든 항은 맵 크기와 무관하도록 0-1 스케일로 맞췄다.
        """

        # 1) 진행(Progress)  ─────────────────────────────────────
        #   이전-거리 − 현재-거리  →  초기-거리 비율로 환산
        dist_now   = np.linalg.norm(self.goal - self.pose[:2])
        progress   = (self.prev_goal_dist - dist_now) / (self.init_dist + 1e-6)
        self.prev_goal_dist = dist_now
        R_goal = 10.0 * progress               # -10 ~ +10 범위

        # 2) 헤딩(Heading)  ──────────────────────────────────────
        theta_err = abs(math.atan2(
            self.goal[1]-self.pose[1], self.goal[0]-self.pose[0]) - self.pose[2])
        theta_err = (theta_err + math.pi) % (2*math.pi) - math.pi  # [-π,π]
        # R_heading = 0.5 * math.cos(theta_err) * (self.v / self.v_max)

        k_cos, k_sin = 0.3, 0.2
        R_heading = (
            k_cos * math.cos(theta_err)            # 정면 정렬
        + k_sin * math.sin(theta_err) * (self.w / self.w_max)  # 올바른 회전 방향
        )

        # 3) 부드러운 속도 패널티  ───────────────────────────────
        target_v  = 0.5 * self.v_max
        R_smooth  = -0.05 * (abs(self.v - target_v) / self.v_max +
                            abs(self.w) / self.w_max)

        # 4) 충돌 위험 패널티  ──────────────────────────────────
        min_dist = float(np.min(self.scan))
        if   min_dist < 0.45:   R_col = -10.0      # 실제 충돌 거리
        elif min_dist < 0.75:   R_col =  -5.0
        elif min_dist < 1.00:   R_col =  -2.0
        else:                   R_col =   0.0

        # 5) 시간 패널티 (지나친 길어진 에피소드 방지)  ───────────
        R_time = -0.01

        return R_goal + R_heading + R_smooth + R_col + R_time
    

    def check_done(self, obs):
        dist_now = np.linalg.norm(self.goal - self.pose[:2])

        # 10 % · 초기거리  —  하지만 최소 1 m
        thr      = max(0.10 * self.init_dist, 1.0)
        success  = dist_now < thr

        collision = np.min(self.scan) < 0.45
        timeout   = self.steps >= self.step_limit
        return success or collision or timeout, success, collision, timeout

    # ----------------------------- step / reset ------------------------------
    def step(self, action):
        # 1) 액션 → 충돌 후 모션 락 적용 ────────────────────────★
        if self.lock_steps > 0:
            self.lock_steps -= 1
            v_cmd = w_cmd = 0.0
        else:
            v_cmd, w_cmd = np.clip(
                action, [-0.2, -self.w_max], [self.v_max, self.w_max])
        tw = Twist();  tw.linear.x = float(v_cmd);  tw.angular.z = float(w_cmd)
        self.cmd_pub.publish(tw)

        self.v, self.w = v_cmd, w_cmd

        # ── 4. 상태 관측 & 보상 ─────────────────────────────────────────
        self.state_ev.wait(); self.state_ev.clear()
        obs    = self.get_obs()
        reward = self.compute_reward(obs)
        done, success, collision, timeout = self.check_done(obs)

        # ── 여기서 성공/충돌 보상 추가 ───────────────────────────
        if success:
            reward += 80.0          # 성공 보상
        elif collision:
            reward -= 30.0           # 충돌 패널티
            self.lock_steps = int(1.0 / self.dt)    # 1초 정지 ───★

        self.steps += 1

        if done:  # 정지
            tw.linear.x = tw.angular.z = 0.0
            self.cmd_pub.publish(tw)

        # ---------------- 에피소드 종료 처리 --------------------------------
        if done:
            # ① 성공 여부 히스토리 -----------------------------
            self.success_buffer.append(bool(success))     # 30-step 커리큘럼용
            self.success_window.append(bool(success))     # 200-step freeze용

            # ② span 확대 조건 (기존 로직 유지)
            if len(self.success_buffer) == self.success_buffer.maxlen:
                success_rate = sum(self.success_buffer) / len(self.success_buffer)
                if success_rate >= self.success_thresh:
                    self.curr_expand_step += 1            # 난이도 ↑
                self.success_buffer.clear()

            # ③ span freeze 조건 (새 로직)
            if len(self.success_window) == self.success_window.maxlen:
                win_rate = sum(self.success_window) / len(self.success_window)
                self.freeze_span = (win_rate < 0.50)      # 60 % 미만이면 일시 중단
                self.success_window.clear()

            # ④ 총 에피소드 수 증가
            self.episode_count += 1


        info = {"success": success,
                "collision": collision,
                "timeout":   timeout}

        return obs, reward, done, info

    def reset(self):
        # 1) 로봇 정지 ───────────────────────────────────────────────
        self.cmd_pub.publish(Twist())
        time.sleep(0.1)

        # 2) 로봇·속도 초기화 ────────────────────────────────────────
        self.reset_robot_pub.publish(
            Pose2D(x=0.0, y=0.0, theta=np.deg2rad(45))
        )
        self.v, self.w = 0.0, 0.0
        time.sleep(0.2)

        # 3) span 계산 (커리큘럼) ────────────────────────────────────
        progress = min(self.curr_expand_step, self.expand_eps) / self.expand_eps
        if not self.freeze_span:
            span = self.start_span + (self.max_span - self.start_span) * (progress ** 0.5)
        else:
            span = getattr(self, "prev_span", self.start_span)
        self.prev_span = span

        # 4) 목표 위치 샘플링 ────────────────────────────────────────
        gx_map = -9.0 + np.random.uniform(0.0, span)
        gy_map = -9.0 + np.random.uniform(0.0, span)
        self.goal = np.array([gx_map, gy_map]) - MAP_ORIGIN
        self.reset_goal_pub.publish(Pose2D(x=gx_map, y=gy_map))
        time.sleep(0.1)

        # 5) 내부 상태 초기화 ────────────────────────────────────────
        self.init_dist = np.linalg.norm(self.goal)          # ─★ 추가
        # step_limit: 500(작은 맵) → 1 000(큰 맵) 선형 보간 ──────★
        d_min, d_max = 4.0, 15.0          # 예상 최소·최대 초기거리
        ratio = np.clip((self.init_dist - d_min) / (d_max - d_min), 0.0, 1.0)
        self.step_limit = int(500 + 500 * ratio)
        # ----------------------------------------------------------
        self.prev_potential = 0.0
        self.prev_goal_dist = self.init_dist                # ─★ 수정
        self.steps          = 0

        return self.get_obs()

