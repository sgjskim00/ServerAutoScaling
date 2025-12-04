import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class ServerAutoScalingEnv(gym.Env):
    """
    [프로젝트 주제: 클라우드 서버 오토스케일링 최적화]
    - 목표: 트래픽을 처리할 수 있는 최소한의 서버만 유지하여 비용 절감
    - State: [현재 트래픽 양, 현재 서버 수]
    - Action: 0(유지), 1(증설), 2(감축)
    - Reward: (안정적인 서비스 보상) - (서버 운영 비용) - (과부하 페널티)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ServerAutoScalingEnv, self).__init__()

        # --- [설정 값] ---
        self.MAX_SERVERS = 20        # 최대 서버 수
        self.MIN_SERVERS = 1         # 최소 서버 수
        self.SERVER_CAPACITY = 1000   # 서버 1대가 처리 가능한 트래픽
        self.COST_PER_SERVER = 10   # 서버 1대당 운영 비용
        self.OVERLOAD_PENALTY = 10.0 # 과부하 발생 시 페널티
        
        # --- [Action Space] ---
        # 0: 유지, 1: 증설(+1), 2: 감축(-1)
        self.action_space = spaces.Discrete(3)

        # --- [Observation Space] ---
        # 관측값: [현재 트래픽(0~2000), 현재 서버 수(0~20)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2000, self.MAX_SERVERS]), 
            dtype=np.float32
        )

        # 초기화
        self.current_step = 0
        self.max_steps = 200 # 에피소드 길이 (24시간 시뮬레이션 등)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 초기 상태: 트래픽 100, 서버 1대
        self.traffic = 100.0
        self.server_count = 1.0
        self.current_step = 0
        
        observation = np.array([self.traffic, self.server_count], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1

        # 1. Action 적용 (서버 증설/감축)
        if action == 1: # Scale Out
            self.server_count = min(self.MAX_SERVERS, self.server_count + 1)
        elif action == 2: # Scale In
            self.server_count = max(self.MIN_SERVERS, self.server_count - 1)
        # action == 0 이면 유지

        # 2. 환경 변화 (트래픽 시뮬레이션)
        # Sine 파형을 이용해 낮에는 높고 밤에는 낮은 트래픽 패턴 생성 + 노이즈
        base_traffic = 1000 + 800 * np.sin(self.current_step * 0.1)
        noise = np.random.normal(0, 50)
        self.traffic = max(0, base_traffic + noise)

        # 3. Reward 계산 (핵심 로직)
        capacity = self.server_count * self.SERVER_CAPACITY
        is_overloaded = self.traffic > capacity
        
        reward = 0
        
        if is_overloaded:
            # 과부하 상태: 페널티 부여 (사용자 불만)
            reward -= self.OVERLOAD_PENALTY
        else:
            # 정상 상태: 서비스 성공 보상
            reward += 2.0
            
        # 서버 비용 차감 (서버가 많을수록 보상 감소 -> 효율성 학습 유도)
        reward -= (self.server_count * self.COST_PER_SERVER)

        # 4. 종료 조건 및 반환
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        observation = np.array([self.traffic, self.server_count], dtype=np.float32)
        info = {"traffic": self.traffic, "servers": self.server_count}
        
        return observation, reward, terminated, truncated, info

# --- [메인 실행 코드] ---
if __name__ == "__main__":
    # 1. 환경 생성 및 검증
    env = ServerAutoScalingEnv()
    check_env(env) # Gymnasium 규격 준수 여부 체크
    print("환경 검증 완료!")

    # 2. 모델 학습 (PPO 알고리즘 사용)
    # 안내문 권장: 논문 결과 재연 또는 문제 해결 [cite: 40, 44]
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    print("학습 시작...")
    model.learn(total_timesteps=10000) # 실험 시 100,000 이상 권장
    print("학습 완료!")

    # 3. 결과 테스트 및 시각화 준비
    obs, _ = env.reset()
    total_reward = 0
    
    # 기록용 리스트
    traffic_history = []
    server_history = []
    
    print("\n[테스트 실행]")
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 기록
        traffic_history.append(info['traffic'])
        server_history.append(info['servers'])

        if terminated or truncated:
            break
            
    print(f"최종 획득 보상: {total_reward:.2f}")

    # 4. 결과 시각화 (보고서용 그래프) [cite: 72]
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(traffic_history, label='Traffic Load', color='blue')
    plt.title('Traffic vs Server Count')
    plt.ylabel('Traffic')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(server_history, label='Active Servers', color='red')
    plt.ylabel('Server Count')
    plt.xlabel('Time Step')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    print("그래프 출력 완료. 보고서에 활용하세요.")