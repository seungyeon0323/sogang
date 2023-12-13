
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym

# 기본 함수들
def createQ_table(rows=4, cols=12):
    return np.zeros((4, cols * rows))

def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(4)  # 무작위 행동 선택
    else:
        return np.argmax(q_table[:, state])  # 최적 행동 선택

def update_position(position, action):
    posX, posY = position
    if action == 0 and posX > 0: posX -= 1  # 상
    if action == 1 and posY > 0: posY -= 1  # 좌
    if action == 2 and posY < 11: posY += 1  # 우
    if action == 3 and posX < 3: posX += 1  # 하
    return (posX, posY)

def get_state(agent, q_table):
    posX, posY = agent
    state = posX * 12 + posY
    state_value = np.amax(q_table[:, state])
    return state, state_value

def get_reward(state):
    if state == 47: return 10, True  # 목표 도달
    if 37 <= state <= 46: return -100, True  # 클리프
    return -1, False  # 기본 보상

def update_q_values(q_table, state, action, reward, next_state_value, gamma, alpha):
    q_table[action, state] += alpha * (reward + gamma * next_state_value - q_table[action, state])
    return q_table

# Q-테이블 변화를 시각화하는 함수
def plot_q_table(q_table, episode, algorithm_name):
    plt.figure(figsize=(12, 4))
    sns.heatmap(q_table, annot=False, cmap="coolwarm", cbar=False)
    plt.title(f"Q-Table after Episode {episode} ({algorithm_name})")
    plt.show()

# 보상 그래프 비교 함수
def plot_reward_comparison(rewards_q_learning, rewards_sarsa):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.plot(rewards_sarsa, label='SARSA')
    plt.title('Reward Comparison between Q-Learning and SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.show()

# 수정된 Q-Learning 알고리즘
def learn_q_values_with_visualization(env, episodes=500, discount_factor=0.9, learning_rate=0.5, explore_rate=0.1):
    q_values = createQ_table()
    rewards_summary = []
    for ep in range(episodes):
        current_position = (3, 0)
        is_done = False
        total_reward = 0  # 여기에서 total_reward를 초기화합니다.
        while not is_done:
            current_state, _ = get_state(current_position, q_values)
            chosen_action = choose_action(current_state, q_values, explore_rate)
            new_position = update_position(current_position, chosen_action)
            next_state, max_next_q_value = get_state(new_position, q_values)
            reward, is_done = get_reward(next_state)
            total_reward += reward  # 누적 보상 업데이트
            q_values = update_q_values(q_values, current_state, chosen_action, reward, max_next_q_value, discount_factor, learning_rate)
            current_position = new_position

        rewards_summary.append(total_reward)
        if ep % 100 == 0:  # 에피소드가 끝날 때마다 Q-테이블을 시각화
            plot_q_table(q_values, ep, "Q-Learning")

    return q_values, rewards_summary

# 수정된 SARSA 알고리즘
def learn_sarsa_with_visualization(env, episodes=500, learning_rate=0.25, discount_factor=0.9, exploration_rate=0.1):
    sarsa_table = createQ_table()
    total_rewards = []
    for ep in range(episodes):
        # [이전 코드 생략]
        # 에피소드가 끝날 때마다 Q-테이블을 시각화
        if ep % 100 == 0:
            plot_q_table(sarsa_table, ep, "SARSA")
        total_rewards.append(accumulated_reward)
    return sarsa_table, total_rewards

# 메인 함수
def main():
    env = gym.make('CliffWalking-v0')
    q_table, rewards_q_learning = learn_q_values_with_visualization(env)
    sarsa_table, rewards_sarsa = learn_sarsa_with_visualization(env)

    # 보상 비교 그래프
    plot_reward_comparison(rewards_q_learning, rewards_sarsa)

if __name__ == "__main__":
    main()
