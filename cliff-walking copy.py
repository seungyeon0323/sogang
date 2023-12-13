# 필요한 패키지를 가져옵니다.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import gymnasium as gym

plt.rcParams['font.family'] = 'AppleGothic.ttf'
plt.rcParams['axes.unicode_minus'] =False

# Q_values (상태-행동 쌍)로 이루어진 테이블을 모두 0으로 초기화합니다.
# Q(s, a)를 임의로 초기화하고, Q(종료 상태, ·) = 0으로 설정합니다.
def createQ_table(rows=4, cols=12):
    """
    Q(s, a) '값 상태-행동 쌍'에 대한 테이블 생성 구현
    
    Args:
        rows -- type(int) 간단한 그리드 월드의 행 수
        cols -- type(int) 간단한 그리드 월드의 열 수
    
    Returns:
        q_table -- type(np.array) 상태-행동 쌍 테이블의 2D 표현
                                     행은 행동이고 열은 상태입니다.
    """
    # 모든 상태 및 행동에 대해 0으로 초기화된 q_table을 생성합니다.
    q_table = np.zeros((4, cols * rows))

    # 빠르게 해당 상태-행동 쌍에 액세스하기 위한 액션 딕셔너리를 정의합니다.
    action_dict = {"UP": q_table[0, :], "LEFT": q_table[1, :], "RIGHT": q_table[2, :], "DOWN": q_table[3, :]}
    
    return q_table


# 정책을 사용하여 액션 선택
# 서튼의 코드 의사 코드: Q에서 파생된 정책을 사용하여 S에서 A를 선택합니다 (예 : ε-탐욕적).
# %10의 탐사를 피하기 위해
def epsilon_greedy_policy(state, q_table, epsilon=0.1):
    """
    Epsilon 탐욕적 정책 구현은 현재 상태와 q_value 테이블을 취해
    Epsilon-탐욕적 정책에 기반하여 어떤 액션을 취할지 결정합니다.
    
    Args:
        epsilon -- type(float) 탐사/이용 비율을 결정합니다.
        state -- type(int) 에이전트의 현재 상태 값 [0:47] 사이
        q_table -- type(np.array) 상태 값 결정
    
    Returns:
        action -- type(int) Q(s, a) 쌍 및 입실론에 기반한 선택된 함수
    """
    # 균일 분포 [0.0, 1.0)에서 무작위 정수를 선택합니다.
    decide_explore_exploit = np.random.random()
    
    if decide_explore_exploit < epsilon:
        action = np.random.choice(4)  # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
    else:
        action = np.argmax(q_table[:, state])  # Q 값 (상태 값)이 가장 큰 액션을 선택합니다.
        
    return action
    

def move_agent(agent, action):
    """
    액션을 기반으로 에이전트를 이동시킵니다.
    
    Args:
        agent -- type(tuple) 그리드 상의 x, y 좌표
        action -- type(int) 에이전트의 위치를 업데이트합니다.
        
    Returns:
        agent -- type(tuple) 에이전트의 새로운 좌표
    """
    # 에이전트의 위치를 얻습니다.
    (posX, posY) = agent
    
    # UP
    if (action == 0) and (posX > 0):
        posX = posX - 1
    # LEFT
    if (action == 1) and (posY > 0):
        posY = posY - 1
    # RIGHT
    if (action == 2) and (posY < 11):
        posY = posY + 1
    # DOWN
    if (action == 3) and (posX < 3):
        posX = posX + 1
    agent = (posX, posY)
    
    return agent


def get_state(agent, q_table):
    """
    에이전트의 위치를 기반으로 상태 및 상태 값 결정
    
    Args:
        agent -- type(tuple) 그리드 상의 x, y 좌표
        q_table -- type(np.array) 상태 값 결정
        
    Returns:
        state -- type(int) [0,47] 사이의 상태 값
        max_state_value -- type(float) 에이전트의 위치에서 최대 상태 값
    """
    # 에이전트의 위치를 얻습니다.
    (posX, posY) = agent
    
    # 상태 값을 얻습니다.
    state = 12 * posX + posY
    
    # 테이블에서 최대 상태 값을 얻습니다.
    state_action = q_table[:, int(state)]
    maximum_state_value = np.amax(state_action)  # 가장 높은 액션에 대한 상태 값을 반환합니다.
    return state, maximum_state_value



def get_reward(state):
    """
    주어진 상태에서 보상을 반환하는 함수
    
    Args:
        state -- type(int) [0,47] 범위 내의 상태 값
        
    Returns: 
        reward -- type(int) 해당 상태에서의 보상 
        game_end -- type(bool) 게임 종료 여부 (클리프에서 떨어지거나 목표에 도달했는지 여부)
    """
    # 게임 계속
    game_end = False
    # 클리프를 제외한 모든 상태는 -1의 값을 갖음
    reward = -1
    # 목표 상태
    if state == 47:
        game_end = True
        reward = 10
    # 클리프
    if 37 <= state <= 46:
        game_end = True
        # 클리프를 만나면 에이전트에게 처벌
        reward = -100

    return reward, game_end

def update_qTable(q_table, state, action, reward, next_state_value, gamma_discount=0.9, alpha=0.5):
    """
    관찰된 보상 및 최대 다음 상태 값에 기반하여 q_table을 업데이트합니다.
    서튼의 책 의사 코드: Q(S, A) <- Q(S, A) + [alpha * (reward + (gamma * maxValue(Q(S', A'))) - Q(S, A)]
    
    Args:
        q_table -- type(np.array) 상태 값을 결정하는 Q 테이블
        state -- type(int) [0,47] 범위 내의 상태 값
        action -- type(int) 액션 값 [0:3] -> [UP, LEFT, RIGHT, DOWN]
        reward -- type(int) 해당 상태에서의 보상 
        next_state_value -- type(float) 다음 상태에서의 최대 상태 값
        gamma_discount -- type(float) 미래 보상의 중요성을 결정하는 할인 계수
        alpha -- type(float) 학습 수렴을 제어하는 값
        
    Returns:
        q_table -- type(np.array) 상태 값을 결정하는 Q 테이블
    """
    update_q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[action, state])
    q_table[action, state] = update_q_value

    return q_table    

def qlearning(env,num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    """
    Q-러닝 알고리즘의 구현. (서튼의 책)
    
    Args:
        num_episodes -- type(int) 에이전트를 훈련할 게임 수
        gamma_discount -- type(float) 미래 보상의 중요성을 결정하는 할인 계수
        alpha -- type(float) 알고리즘이 수렴하는 속도 (상태를 빠르게 또는 느리게 업데이트하는 것으로 생각할 수 있음)
        epsilon -- type(float) 탐험/이용 비율 (예: 기본값 0.1은 10%의 탐험을 의미)
        
    Returns:
        q_table -- type(np.array) 상태 값을 결정하는 Q 테이블
        reward_cache -- type(list) 누적 보상을 포함하는 리스트
    """
    # 모든 상태를 0으로 초기화
    # 종료 상태인 클리프 건너기
    reward_cache = list()
    step_cache = list()
    q_table = createQ_table()
    agent = (3, 0)  # 왼쪽 하단에서 시작
    # 에피소드를 반복
    for episode in range(0, num_episodes):
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        agent = (3, 0)  # 왼쪽 하단에서 시작
        game_end = False
        reward_cum = 0  # 에피소드의 누적 보상
        step_cum = 0  # 게임 종료까지의 반복 횟수
        while not game_end:
            # 에이전트의 위치에서 상태를 가져옴
            state, _ = get_state(agent, q_table)
            # epsilon-탐욕 정책을 사용하여 행동 선택
            action = epsilon_greedy_policy(state, q_table)
            # 에이전트를 다음 상태로 이동
            agent = move_agent(agent, action)
            step_cum += 1
            env = visited_env(agent, env)  # 방문한 경로 표시
            # 다음 상태 값 관찰
            next_state, max_next_state_value = get_state(agent, q_table)
            # 보상 관찰 및 게임 종료 여부 결정
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            # Q 테이블 업데이트
            q_table = update_qTable(q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)
            # 상태 업데이트
            state = next_state
        reward_cache.append(reward_cum)
        if episode > 498:
            print("Q-러닝으로 훈련된 에이전트 (500번 반복 후)")
            print(env)  # 에이전트가 취한 마지막 2개 경로를 표시
        step_cache.append(step_cum)
    return q_table, reward_cache, step_cache



def sarsa(env, num_episodes=500, alpha=0.25, gamma_discount=0.9, epsilon=0.1):
    # 코드 내용 유지

    """
    SARSA 알고리즘의 구현. (서튼의 책)
    
    Args:
        num_episodes -- type(int) 에이전트를 훈련할 게임 수
        gamma_discount -- type(float) 미래 보상의 중요성을 결정하는 할인 계수
        alpha -- type(float) 알고리즘이 수렴하는 속도 (상태를 빠르게 또는 느리게 업데이트하는 것으로 생각할 수 있음)
        epsilon -- type(float) 탐험/이용 비율 (예: 기본값 0.1은 10%의 탐험을 의미)
        
    Returns:
        q_table -- type(np.array) 상태 값을 결정하는 Q 테이블
        reward_cache -- type(list) 누적 보상을 포함하는 리스트
        step_cache -- type(list) 게임 종료까지의 반복 횟수를 포함하는 리스트
    """
    # 모든 상태를 0으로 초기화
    # 종료 상태인 클리프 건너기
    q_table = createQ_table()
    step_cache = list()
    reward_cache = list()
    # 에피소드를 반복
    for episode in range(0, num_episodes):
        agent = (3, 0)  # 왼쪽 하단에서 시작
        game_end = False
        reward_cum = 0  # 에피소드의 누적 보상
        step_cum = 0  # 게임 종료까지의 반복 횟수
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        # 정책을 사용하여 액션 선택
        state, _ = get_state(agent, q_table)
        action = epsilon_greedy_policy(state, q_table)
        while not game_end:
            # 에이전트를 다음 상태로 이동
            agent = move_agent(agent, action)
            env = visited_env(agent, env)
            step_cum += 1
            # 다음 상태 값 관찰
            next_state, _ = get_state(agent, q_table)
            # 보상 관찰 및 게임 종료 여부 결정
            reward, game_end = get_reward(next_state)
            reward_cum += reward 
            # 정책 및 다음 상태를 사용하여 다음 액션 선택
            next_action = epsilon_greedy_policy(next_state, q_table)
            # Q 테이블 업데이트
            next_state_value = q_table[next_action][next_state]  # q-러닝과 달리 다음 액션은 정책에 따라 결정됨
            q_table = update_qTable(q_table, state, action, reward, next_state_value, gamma_discount, alpha)
            # 상태 및 액션 업데이트
            state = next_state
            action = next_action  # q-러닝과 달리 상태 및 액션 모두 업데이트 필요
        reward_cache.append(reward_cum)
        step_cache.append(step_cum)
        if episode > 498:
            print("SARSA로 훈련된 에이전트 (500번 반복 후)")
            print(env)  # 에이전트가 취한 마지막 2개 경로를 표시 
    return q_table, reward_cache, step_cache


def visited_env(agent, env):
    """
    에이전트가 따라가는 경로를 시각화합니다.
    
    Args:
        agent -- type(tuple) 현재 에이전트의 위치 (행, 열)
        env -- type(np.array) 환경을 나타내는 2D 배열
    """
    (posY, posX) = agent
    env[posY][posX] = 1
    return env
    
    
def retrieve_environment(q_table, action):
    """
    특정 액션에 대한 환경 상태 값을 표시합니다.
    디버그 목적으로 구현되었습니다.
    
    Args:
        q_table -- type(np.array) 상태 값을 결정하는 Q 테이블
        action -- type(int) 액션 값 [0:3] -> [UP, LEFT, RIGHT, DOWN]
    """
    env = q_table[action, :].reshape((4, 12))
    print(env)  # 환경 값 표시
    
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
            # 샘플을 정규화합니다.
            normalized_reward = (cur_reward - rewards_mean) / rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    # 그래프를 준비합니다.    
    plt.plot(cum_rewards_q, label="q_learning")
    plt.plot(cum_rewards_SARSA, label="SARSA")
    plt.ylabel('누적 보상')
    plt.xlabel('에피소드 배치 (샘플 크기 10) ')
    plt.title("Q-Learning/SARSA 누적 보상의 수렴")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show() 

    
def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    """
        에피소드 당 걸린 단계 수를 시각화합니다.
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
    count = 0  # 배치를 결정하는 데 사용됩니다
    cur_step = 0  # 배치에 대한 누적 단계
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if count == 10:
            # 샘플을 정규화합니다.
            normalized_step = (cur_step - steps_mean) / steps_std
            cum_step_SARSA.append(normalized_step)
            cur_step = 0
            count = 0      
    # 그래프를 준비합니다.    
    plt.plot(cum_step_q, label="q_learning")
    plt.plot(cum_step_SARSA, label="SARSA")
    plt.ylabel('반복 횟수')
    plt.xlabel('에피소드 배치 (샘플 크기 10) ')
    plt.title("Q-Learning/SARSA 게임 종료까지 걸리는 반복 횟수")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


    
def plot_qlearning_smooth(reward_cache):
    """
    Visualizes the reward convergence using weighted average of previous 10 cumulative rewards
    NOTE: Normalization gives better visualization
    
    Args:
        reward_cache -- type(list) contains cumulative_rewards for episodes
    """
    mean_rev = (np.array(reward_cache[0:11]).sum())/10
    # initialize with cache mean
    cum_rewards = [mean_rev] * 10
    idx = 0
    for cache in reward_cache:
        cum_rewards[idx] = cache
        idx += 1
        smooth_reward = (np.array(cum_rewards).mean())
        cum_rewards.append(smooth_reward)
        if(idx == 10):
            idx = 0
        
    plt.plot(cum_rewards)
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning  Convergence of Cumulative Reward")
    plt.legend(loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def generate_heatmap(q_table):
    """
        Generates heatmap to visualize agent's learned actions on the environment
    """
    import seaborn as sns; sns.set()
    # display mean of environment values using a heatmap
    data = np.mean(q_table, axis = 0)
    print(data)
    data = data.reshape((4, 12))
    ax = sns.heatmap(np.array(data))
    return ax
    
def main():
    # Gym 환경 생성
    env = gym.make('CliffWalking-v0')

    # 500 에피소드 동안 상태 변화를 학습하고 누적 보상을 얻습니다.
    # SARSA
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa(env)
    # QLEARNING
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = qlearning(env)
    plot_number_steps(step_cache_qlearning, step_cache_SARSA)
    # 결과 시각화
    plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA)

    # 히트맵 생성
    print("Q-learning 환경 시각화")
    ax_q = generate_heatmap(q_table_qlearning)
    print(ax_q)

    print("SARSA 환경 시각화")
    ax_SARSA = generate_heatmap(q_table_SARSA)
    print(ax_SARSA)

    # 환경에 대한 정보를 제공하는 디버그 메서드
    want_to_see_env = False
    if want_to_see_env:
        print("UP")
        retrieve_environment(q_table_qlearning, 0)
        print("LEFT")
        retrieve_environment(q_table_qlearning, 1)
        print("RIGHT")
        retrieve_environment(q_table_qlearning, 2)
        print("DOWN")
        retrieve_environment(q_table_qlearning, 3)

    want_to_see_env = False
    if want_to_see_env:
        print("UP")
        retrieve_environment(q_table_SARSA, 0)

        print("LEFT")
        retrieve_environment(q_table_SARSA, 1)
        print("RIGHT")
        retrieve_environment(q_table_SARSA, 2)
        print("DOWN")
        retrieve_environment(q_table_SARSA, 3)

if __name__ == "__main__":
    main()
