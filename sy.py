import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
import sys
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from collections import defaultdict
#from lib import plotting


plt.style.use('ggplot')

# CliffWalking 환경을 생성합니다.
env = CliffWalkingEnv()

# 환경을 시각화합니다.
env.render()

# 가능한 행동을 나타내는 리스트를 정의합니다.
action = ["up", "right", "down", "left"]

# 환경의 상태 수를 출력합니다.
print("상태의 수:", env.nS)
# 에이전트가 취할 수 있는 행동 수를 출력합니다.
print("에이전트가 취할 수 있는 행동의 수:", env.nA)

# 현재 위치한 상태를 출력합니다.
print("현재 상태:", env.s)
# 가능한 행동에 대한 전이 확률을 출력합니다.
print("현재 상태에서의 전이 확률:", env.P[env.s])

# 무작위로 행동을 선택하여 한 단계를 실행합니다.
# 다음 상태, 보상, 종료 여부, 전이 확률을 출력합니다.
rnd_action = random.randint(0, 3)
print("선택한 행동:", action[rnd_action])
next_state, reward, is_terminal, t_prob = env.step(rnd_action)
print("전이 확률:", t_prob)
print("다음 상태:", next_state)
print("받은 보상:", reward)
print("종료 상태:", is_terminal)

# 업데이트된 상태를 시각화합니다.
env.render()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Q 함수와 epsilon에 기반한 epsilon-greedy 정책을 생성합니다.

    Args:
        Q: 상태(state)를 행동(action) 값으로 매핑한 딕셔너리.
           각 값은 길이가 nA인 numpy 배열입니다 (아래 참조).
        epsilon: 0에서 1 사이의 값으로, 무작위 행동을 선택할 확률입니다.
        nA: 환경에서의 행동 수.

    Returns:
        주어진 관측값을 인수로 받아 각 행동에 대한 확률을 나타내는 numpy 배열을 반환하는 함수입니다.
    """
    def policy_fn(observation):
        # 모든 행동에 대한 확률을 초기화합니다.
        pi = np.ones(nA, dtype=float) * (epsilon / nA)
        # Q 함수에서 최적의 행동을 찾습니다.
        best_action = np.argmax(Q[observation])
        # 최적의 행동에 대한 확률을 업데이트합니다.
        pi[best_action] += (1.0 - epsilon)
        return pi
    
    return policy_fn

# 샘플 정책에 따라 클리프워킹 환경에서 에피소드 생성
def generate_episode(policy, verbose=False):
    episode = []
    env = CliffWalkingEnv()
    curr_state = env.reset()
    # 현재 상태에서의 행동 확률을 샘플링
    probs = policy(curr_state)
    action = np.random.choice(np.arange(len(probs)), p=probs)
    
    while True:
        if verbose:
            print("현재 관측값:")
            print("현재 위치:", curr_state)
            #print (env.render())
        
        # 선택한 행동을 적용하여 다음 관측값, 보상 등을 얻음
        next_obs, reward, is_terminal, _ = env.step(action)
        
        if verbose:
            print("선택한 행동:", action)
            print("다음 관측값:", next_obs)
            print("받은 보상:", reward)
            print("종료 상태:", is_terminal)
            #print (env.render())
            print("-" * 20)
        
        # 에피소드에 현재 상태, 행동, 보상 추가
        episode.append((curr_state, action, reward))
        
        # 다음 행동을 선택
        next_probs = policy(next_state)
        next_action = np.random.choice(np.arange(len(next_probs)), p=next_probs)
    
        curr_state = next_obs
        action = next_action

        if is_terminal:
            break

    return episode
    
# Q 함수 및 샘플 정책 생성
Q = defaultdict(lambda: np.zeros(env.action_space.n))
policy = make_epsilon_greedy_policy(Q, 0.1, env.action_space.n)
# 에피소드 생성 및 길이 출력
e = generate_episode(policy)
#print ("Episode:", e)
print("에피소드 길이:", len(e))


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA 알고리즘: 온-폴리시 TD 제어. 최적의 엡실론-그리디 정책을 찾습니다.

    Args:
        env: OpenAI 환경.
        num_episodes: 실행할 에피소드 수.
        discount_factor: 감마 할인 계수.
        alpha: TD 학습률.
        epsilon: 무작위 행동을 샘플링할 확률. 0과 1 사이의 실수.

    Returns:
        (Q, stats) 튜플.
        Q는 최적의 행동 가치 함수로, 상태를 행동 가치로 매핑하는 딕셔너리입니다.
        stats는 에피소드 길이와 에피소드 보상을 위한 두 개의 NumPy 배열을 포함하는 EpisodeStats 객체입니다.
    """

    # 최종 행동 가치 함수.
    # 상태를 (행동 -> 행동 가치)로 매핑하는 중첩된 딕셔너리입니다.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # 유용한 통계를 추적합니다.
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # 따르는 정책
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # 디버깅에 유용한 현재 에피소드를 출력합니다.
        if (i_episode + 1) % 100 == 0:
            print("\r에피소드 {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # 환경을 재설정하고 첫 번째 행동을 선택합니다.
        state = env.reset()
        probs = policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)

        # 방법: 1
        # 실행에 많은 시간이 걸립니다.
        # ep = generate_episode(policy)
        # for i in range(len(ep)-1):
        #     state, action, reward = ep[i]
        #     next_state, next_action, next_reward = ep[i+1]
        #
        #     td_target = reward + discount_factor * Q[next_state][next_action]
        #     td_error = td_target - Q[state][action]
        #     Q[state][action] += alpha * td_error
        #
        #     stats.episode_rewards[i_episode] += reward
        #     stats.episode_lengths[i_episode] = t

        # 방법: 2
        # 환경에서 한 단계
        for t in itertools.count():
            # 한 단계를 밟습니다.
            next_state, reward, is_terminal, _ = env.step(action)

            # 다음 행동을 선택합니다.
            next_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_probs)), p=next_probs)

            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if is_terminal:
                break

            state = next_state
            action = next_action

    return Q, stats


Q, stats = sarsa(env, 200)

plotting.plot_episode_stats(stats)

# get greedy policy from Q
policy = np.array([np.argmax(Q[key]) if key in Q else -1 for key in np.arange(48)])
# get value function from Q using greedy policy
v = ([np.max(Q[key]) if key in Q else 0 for key in np.arange(48)])

print("Reshaped Grid Policy:")
actions = np.stack([action for _ in range(len(policy))], axis=0)
print (np.reshape(policy, (4, 12)))
print ("")

print ("Optimal Policy:")
print(np.take(actions, np.reshape(policy, (4, 12))))
print("")

print("Optimal Value Function:")
print(np.reshape(v, (4, 12)))
print("")