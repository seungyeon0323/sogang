import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import warnings
from matplotlib import rc 

rc('font', family='AppleGothic') 			
plt.rcParams['axes.unicode_minus'] = False  

# 유틸리티 함수들
def initialize_q_table(rows=4, cols=12):
    """ Q-테이블을 0으로 초기화하는 함수 """
    return np.zeros((4, cols * rows))

def select_action(state, q_table, epsilon):
    """ Epsilon-탐욕 전략을 통해 행동을 선택하는 함수 """
    return np.random.choice(4) if np.random.random() < epsilon else np.argmax(q_table[:, state])

def move_agent(agent_position, action):
    """ 에이전트의 위치를 업데이트하는 함수 """
    posX, posY = agent_position
    # 상, 좌, 우, 하에 따라 위치를 업데이트
    if action == 0 and posX > 0: posX -= 1 # 상
    if action == 1 and posY > 0: posY -= 1 # 하
    if action == 2 and posY < 11: posY += 1 # 좌
    if action == 3 and posX < 3: posX += 1 # 우
    return (posX, posY)

def calculate_state(agent, q_table):
    """ 에이전트의 현재 상태와 Q-테이블에서의 값을 계산하는 함수 """
    posX, posY = agent
    state = posX * 12 + posY
    state_value = np.amax(q_table[:, state])
    return state, state_value

def determine_reward(state):
    """ 현재 상태에 따른 보상을 결정하는 함수 """
    # 목표에 도달하면 보상 10, 클리프면 -100, 그 외에는 -1의 보상
    if state == 47: return 10, True
    if 37 <= state <= 46: return -100, True
    return -1, False

def adjust_q_values(q_table, state, action, reward, next_state_val, gamma, alpha):
    """ 벨만 방정식에 따라 Q-값을 업데이트하는 함수 """
    q_table[action, state] += alpha * (reward + gamma * next_state_val - q_table[action, state])
    return q_table

# 시각화 함수들
def visualize_q_table(q_table, episode, algo_name):
    """ Q-테이블을 히트맵으로 시각화하는 함수 """
    plt.figure(figsize=(12, 4))
    sns.heatmap(q_table, annot=False, cmap="coolwarm", cbar=False)
    plt.title(f"에피소드 {episode} 후 Q-테이블 ({algo_name})")
    plt.show()

def compare_rewards(rewards_q_learn, rewards_sarsa_learn):
    """ Q-Learning과 SARSA 알고리즘의 보상을 비교하는 함수 """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_q_learn, label='Q-Learning', color='orange')
    plt.plot(rewards_sarsa_learn, label='SARSA', color='blue')
    plt.title('Q-Learning/SARSA 게임 중 누적된 보상 비교 그래프')
    plt.xlabel('에피소드 번호 (총반복 횟수 10)')
    plt.ylabel('누적 보상')
    plt.legend()
    plt.show()


# Q-Learning과 SARSA 함수는 위에 정의한 함수들을 사용하여 구현

def run_q_learning(env, episodes=500, learn_rate=0.5, discount=0.9, explore_rate=0.1):
    """
    env: 환경 객체
    episodes: 실행할 에피소드의 수
    learn_rate: 학습률 (Q 값을 업데이트할 때 얼마나 빠르게 학습할지)
    discount: 할인율 (미래 보상을 현재 가치로 얼마나 할인할지)
    explore_rate: 탐색률 (무작위 행동을 선택할 확률)
    """
    q_vals = initialize_q_table()  # Q-테이블 초기화
    rewards_sum = []  # 각 에피소드별 총 보상을 저장할 리스트
    
    for ep in range(episodes):
        position = (3, 0)  # 시작 위치 설정
        done = False  # 에피소드 종료 여부
        total_reward = 0  # 에피소드별 총 보상

        while not done:
            state, _ = calculate_state(position, q_vals)  # 현재 상태 계산
            action = select_action(state, q_vals, explore_rate)  # 행동 선택
            new_pos = move_agent(position, action)  # 에이전트 이동
            next_state, max_next_q = calculate_state(new_pos, q_vals)  # 다음 상태 계산
            reward, done = determine_reward(next_state)  # 보상 및 종료 여부 결정
            total_reward += reward  # 총 보상 업데이트
            # Q-테이블 업데이트
            q_vals = adjust_q_values(q_vals, state, action, reward, max_next_q, discount, learn_rate)
            position = new_pos  # 에이전트 위치 업데이트
        
        rewards_sum.append(total_reward)  # 총 보상을 리스트에 추가

        # 매 100 에피소드마다 Q-테이블 시각화
        if ep % 100 == 0:
            visualize_q_table(q_vals, ep, "Q-Learning")
    
    return q_vals, rewards_sum  # Q-테이블과 보상 리스트 반환

def run_sarsa(env, episodes=500, learn_rate=0.25, discount=0.9, explore_rate=0.1):
    """
    SARSA 알고리즘을 실행하는 함수입니다.
    매개변수는 Q-Learning과 동일합니다.
    """
    sarsa_vals = initialize_q_table()  # SARSA 테이블 초기화
    total_rewards = []  # 총 보상 리스트

    for ep in range(episodes):
        position = (3, 0)  # 시작 위치
        state, _ = calculate_state(position, sarsa_vals)  # 현재 상태 계산
        action = select_action(state, sarsa_vals, explore_rate)  # 행동 선택
        done = False  # 에피소드 종료 여부
        accumulated_reward = 0  # 누적 보상

        while not done:
            new_pos = move_agent(position, action)  # 에이전트 이동
            next_state, _ = calculate_state(new_pos, sarsa_vals)  # 다음 상태
            next_action = select_action(next_state, sarsa_vals, explore_rate)  # 다음 행동 선택
            reward, done = determine_reward(next_state)  # 보상 및 종료 여부 결정
            accumulated_reward += reward  # 누적 보상 업데이트
            # SARSA 테이블 업데이트
            sarsa_vals = adjust_q_values(sarsa_vals, state, action, reward, sarsa_vals[next_action, next_state], discount, learn_rate)
            # 에이전트 위치 및 상태 업데이트
            position, state, action = new_pos, next_state, next_action
        
        total_rewards.append(accumulated_reward)  # 누적 보상을 리스트에 추가

        # 매 100 에피소드마다 SARSA 테이블 시각화
        if ep % 100 == 0:
            visualize_q_table(sarsa_vals, ep, "SARSA")
    
    return sarsa_vals, total_rewards  # SARSA 테이블과 보상 리스트 반환

def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    """
    보상 수렴을 시각화합니다.
    
    Args:
        reward_cache -- type(list) 누적 보상을 포함하는 리스트
    """
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0  # 배치를 결정하는 데 사용됩니다
    cur_reward = 0  # 배치에 대한 누적 보상
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if count == 10:
            # 샘플을 정규화합니다.
            normalized_reward = (cur_reward - rewards_mean) / rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    count = 0  # 배치를 결정하는 데 사용됩니다
    cur_reward = 0  # 배치에 대한 누적 보상
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if count == 10:
            # 샘플을 정규화
            normalized_reward = (cur_reward - rewards_mean) / rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    # 그래프를 준비
    plt.plot(cum_rewards_q, label="q_learning")
    plt.plot(cum_rewards_SARSA, label="SARSA")
    plt.ylabel('누적 보상')
    plt.xlabel('에피소드 배치 (샘플 크기 10) ')
    plt.title("Q-Learning/SARSA 누적 보상의 수렴")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show() 


def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    """
        에피소드 당 걸린 단계 수를 시각화
    """    
    cum_step_q = []
    steps_mean = np.array(step_cache_qlearning).mean()
    steps_std = np.array(step_cache_qlearning).std()
    count = 0  # 배치를 결정하는 데 사용됩니다
    cur_step = 0  # 배치에 대한 누적 단계
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if count == 10:
            # 샘플을 정규화합니다.
            normalized_step = (cur_step - steps_mean) / steps_std
            cum_step_q.append(normalized_step)
            cur_step = 0
            count = 0
            
    cum_step_SARSA = []
    steps_mean = np.array(step_cache_SARSA).mean()
    steps_std = np.array(step_cache_SARSA).std()
    count = 0  # 배치를 결정하는 데 사용
    cur_step = 0  # 배치에 대한 누적 단계
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if count == 10:
            # 샘플을 정규화
            normalized_step = (cur_step - steps_mean) / steps_std
            cum_step_SARSA.append(normalized_step)
            cur_step = 0
            count = 0      
    # 그래프를 준비
    plt.plot(cum_step_q, label="q_learning")
    plt.plot(cum_step_SARSA, label="SARSA")
    plt.ylabel('반복 횟수')
    plt.xlabel('에피소드 배치 (샘플 크기 10) ')
    plt.title("Q-Learning/SARSA 게임 종료까지 걸리는 반복 횟수")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


# 메인 함수
def main():
    env = gym.make('CliffWalking-v0')  # CliffWalking 환경 초기화
    q_learn_table, q_learn_rewards = run_q_learning(env)  # Q-Learning 알고리즘 실행
    sarsa_table, sarsa_rewards = run_sarsa(env)  # SARSA 알고리즘 실행

    compare_rewards(q_learn_rewards, sarsa_rewards)  # 보상 비교 그래프 출력

if __name__ == "__main__":
    main()
