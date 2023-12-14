# cliff walking
  강화학습 기초 프로젝트

## 💻 프로젝트 소개
* Cliff Walking 문제는 강화 학습 분야에서 제가 적용할 수 있을것이라고 판단하여 선택하게 되었습니다. 
이 프로젝트를 통해, 강화 학습 알고리즘의 기본 개념을 실제로 적용해보고, 그 효과와 한계를 직접 실험하며 이해하고자하였습니다.

### ⏰ 프로젝트 기간
* 2023-10-31 ~ 2023-12-14

### 👩🏻‍💻 프로젝트 멤버
* 유승연(A69034)

### 💿 개발환경
- gym.make('CliffWalking-v0’)
- python 
- Numpy, maplotlib, seaborn, gymnasium

### 🔖하이퍼파라미터
- env: gym 라이브러리의 CliffWalking 환경을 사용
- episodes: 실행할 총 에피소드의 수(500)
- learn_rate: 학습률로, Q 값을 얼마나 빠르게 업데이트할지 (0.5)
- discount: 할인율로, 미래 보상을 현재 가치로 얼마나 할인할지 (0.9)
- explore_rate: 탐색률로, 에이전트가 무작위 행동을 선택할지 (1.0)
  
## 📌 주요기능

### action
  
```
  def move_agent(agent_position, action): """ 에이전트의 위치를 업데이트하는 함수 """
    posX, posY = agent_position
    ''' 상, 좌, 우, 하에 따라 위치를 업데이트'''
    if action == 0 and posX > 0: posX -= 1 # 상
    if action == 1 and posY > 0: posY -= 1 # 하
    if action == 2 and posY < 11: posY += 1 # 좌
    if action == 3 and posX < 3: posX += 1 # 우
    return (posX, posY)
```
### reward
```
  def determine_reward(state):
    """ 현재 상태에 따른 보상을 결정하는 함수 """
    # 목표에 도달하면 보상 10, 클리프면 -100, 그 외에는 -1의 보상
    if state == 47: return 10, True  # 목표 지점에 도달했을 때
    if 37 <= state <= 46: return -100, True  # 클리프에 떨어졌을 때
    return -1, False  # 그 외의 이동에 대해서는 -1의 보상
```
함수의 내용을 자세히 살펴보면:

- if state == 47: - 상태 47은 CliffWalking 환경에서 목표 지점에 해당합니다. 에이전트가 목표에 도달하면, 보상으로 10을 받고, True를 함께 반환하여 에피소드가 종료되었음을 나타냄</br>if 37 <= state <= 46: - 이 범위 내의 상태들은 클리프를 나타냄 에이전트가 클리프에 떨어지면 매우 큰 음수 값인 -100의 보상을 받고, 에피소드는 역시 True를 반환하여 종료
 이는 에이전트가 클리프를 피하도록 매우 강력한 음의 인센티브를 제공

- return -1, False - 위의 두 조건에 해당하지 않는 모든 다른 이동에 대해서는 -1의 보상을 받음 이는 에이전트가 목표 지점에 더 빠르게 도달하도록 유도하기 위한 것으로, 에이전트가 불필요한 이동을 줄이고 더 효율적인 경로를 찾도록 동기를 부여하며 False는 에피소드가 아직 종료되지 않았음을 나타냄

- 이러한 보상 체계는 에이전트가 안전하게 목표 지점에 도달하기 위한 경로를 학습하면서 위험을 회피하는 전략을 개발하도록 유도하는 데 중요 또한 이는 절벽으로 인한 큰 음수 보상은 위험한 행동에 대한 강력한 처벌로 작용하여, 에이전트가 해당 행동을 피하게 만드는 반면, 목표에 도달했을 때의 긍정적인 보상은 에이전트가 목표를 향해 이동

### Q - learning
```
def run_q_learning(env, episodes=500, learn_rate=0.5, discount=0.9, explore_rate=0.1):

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
```

- q_vals: Q-테이블을 초기화. 
- rewards_sum: 각 에피소드별 총 보상을 저장할 리스트를 초기화.
- 에피소드 반복:
</br>for ep in range(episodes): 지정된 에피소드 수만큼 반복을 실행
- 에이전트의 행동:
</br>while not done: 에피소드가 끝날 때까지 (done이 True가 될 때까지) 반복
- 현재 상태 계산:
</br>state, _ = calculate_state(position, q_vals): 현재 에이전트의 위치(position)를 기반으로 상태를 계산
</br>action = select_action(state, q_vals, explore_rate): 현재 상태에 기반하여 행동을 선택 이때 explore_rate에 따라 무작위 행동이 선택될 수도 있음
</br>new_pos = move_agent(position, action): 선택된 행동을 기반으로 에이전트의 위치를 업데이트
</br>next_state, max_next_q = calculate_state(new_pos, q_vals): 새 위치에서의 다음 상태와 그 상태의 최대 Q 값을 계산
</br>reward, done = determine_reward(next_state): 새로운 상태에 대한 보상을 받고, 에피소드가 종료되었는지 여부를 결정
</br>total_reward += reward: 총 보상에 현재 보상을 합산
</br>q_vals = adjust_q_values(q_vals, state, action, reward, max_next_q, discount, learn_rate): 벨만 방정식을 사용하여 Q-테이블을 업데이트
</br>position = new_pos: 에이전트의 위치를 새 위치로 업데이트 
- ε-탐욕적 정책 (Epsilon-Greedy Policy): 에이전트가 탐험(새로운 행동 시도)과 이용(현재 정보를 기반으로 최적의 행동 선택) 사이에서 균형을 맞추도록 도움

### sarsa

```
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

```

- q_vals: Q-테이블을 초기화. 
- rewards_sum: 각 에피소드별 총 보상을 저장할 리스트를 초기화 
- 에피소드 내 반복:
</br>for ep in range(episodes): 지정된 에피소드 수만큼 반복을 실행.
</br>while not done: 에피소드가 끝날 때까지 (done이 True가 될 때까지) 반복합니다.
- 현재 상태 계산:
</br>new_pos = move_agent(position, action): 선택된 행동을 기반으로 에이전트의 위치를 업데이트.
</br>next_state, _ = calculate_state(new_pos, sarsa_vals): 새 위치에서의 다음 상태를 계산
</br>next_action = select_action(next_state, sarsa_vals, explore_rate): 다음 상태에 대한 행동을 선택 이는 SARSA가 on-policy 알고리즘임을 반영
</br>reward, done = determine_reward(next_state): 새로운 상태에 대한 보상을 받고, 에피소드가 종료되었는지 여부를 결정
</br>accumulated_reward += reward: 누적 보상에 현재 보상을 합산
- ε-탐욕적 정책 (Epsilon-Greedy Policy):이 정책은 탐험과 이용의 균형을 맞추는 데 사용. ε 확률로 임의의 행동을 선택(탐험)하고, 1-ε 확률로 현재 Q-테이블에 기반하여 최적의 행동을 선택(이용)

## 🔥 히트맵

### Q-learning

<img width="866" alt="image" src="https://github.com/seungyeon0323/sogang/assets/153755354/04674099-960c-494c-b708-48bf5584a784">

- 초기 에피소드 (에피소드 0): 히트맵이 대체로 균일합니다. 이는 Q-테이블의 Q값이 학습을 통해 아직 업데이트되지 않았기 때문에 처음에는 모든 값이 동일하게 시작됨을 나타냅니다.</br>

- 중간 에피소드 (에피소드 100, 200 ): 에피소드가 진행됨에 따라 히트맵에 다양한 색상과 다양한 Q값을 나타내며, 일반적으로 어두운 색상은 낮은 Q값을, 밝은 색상은 높은 Q값을 나타냅니다. 이는 알고리즘이 받은 보상을 기반으로 다양한 상태에서 어떤 행동이 더 나은지를 학습하고 있음을 보여줍니다. </br>
- 후기 에피소드 (에피소드 300, 400): 이상적으로 학습이 진행됨에 따라 히트맵에서 더 명확한 패턴이 형성되어갑니다. 이는 Q-Learning 알고리즘이 최적의 정책으로 수렴하고 있으며, 장기적으로 가장 보상이 높은 행동을 학습하고 있음을 나타냅니다.

### Sarsa

<img width="861" alt="image" src="https://github.com/seungyeon0323/sogang/assets/153755354/f553fbca-9a20-482b-ab0e-91e9180af925"></br>
- 초기 에피소드 (에피소드 0): 히트맵이 대체로 균일합니다.  Q-Learning과 대체로 비슷합니다.</br>
하지만 알 수 있는 것은 SARSA 알고리즘은 탐색 중 실제로 취할 다음 행동을 고려한다는 것을 알 수 있습니다.
- 중간 에피소드(100, 200 ):더 많은 학습이 진행되면서시작 지점과 목표 지점 주변에서 Q-값의 변화가 생기며 절벽을 피하는 경로가 더욱 분명해지고, 이는 더 밝은 색상으로 나타납니다.
- 후기 에피소드 (에피소드 300, 400): 학습이 더 진행되면서, 에이전트는 절벽 근처에서 낮은 Q-값을 유지하며 안전한 경로를 학습합니다. 
</br>이 시점에서는 최적 경로가 더 분명하게 나타나며, 이 경로는 히트맵에서 밝은 색상의 연속으로 나타납니다.

</br> -> 히트맵을 통하여 알고리즘이 에피소드를 거치면서 어떻게 상태 공간을 탐색하는지를 시각적으로 비교할 수 있었습니다.

## 🕶️ 시각화
<img width="419" alt="image" src="https://github.com/seungyeon0323/sogang/assets/153755354/833f890b-4af6-4fc9-96ac-2e8dd1669065"></br>
- 해당 그래프를 보게되면 초기에는 두 알고리즘 모두 큰 손실을 보이는 것처럼 보이고 있습니다.
</br>그러나 에피소드가 진행됨에 따라, 손실의 크기가 줄어들고, 보상이 안정화되는 되어가고 있습니다.
 </br>-> 알고리즘이 학습을 통해 환경에 적응하고, 더 나은 policy를 찾아가고 있음을 나타냅니다.

- Q-Learning(노란색 선)과 SARSA(파란색 선) 알고리즘은 에피소드가 진행됨에 따라 보상의 변동성이 많이 감소해가고 있습니다.
  </br>그리고 거의 두 선이 겹치는 부분이 많아 두 알고리즘의 성능이 상당히 유사하다는 것을 알 수 있습니다. 

- 그러나 에피소드가 진행될수록 Q-Learning이 더 높은 보상을 안정적으로 유지하는 것을 알 수가 있는데요.
</br>이는 Q-Learning이 최적의 정책을 찾는 데 더 효과적일 수 있음을 나타냅니다.



## 마무리하며 😁
###  프로젝트 후 느낀점
- Q-Learning과 SARSA 알고리즘의 학습 과정과 성능을 시각화함으로써 두 알고리즘이 어떻게 다른지, 어떤 상황에서 한 알고리즘이 다른 알고리즘보다 나은 성능을 보이는지를 이해할 수 있습니다
 시각화를 통하여 알고리즘의 안정성을 확인할 수 있었고 직관적으로 분석이 가능하다는 것을 느꼈습니다.
- Q-Learning이 더 빠르게 최적의 경로를 찾아가는 경향이 있는 반면, SARSA는 현재 정책에 더 의존하면서 조심스럽게 탐색한다는 것을 직접 실습을 통해 알 수 있었습니다.


### 프로젝트 개선점
- 탐색률을 고정된 값이 아닌 에피소드가 진행됨에 따라 점점 감소하는 방식으로 설정하여 초기에는 탐색을 더 많이 하고, 시간이 지남에 따라 최적의 정책에 수렴하도록 할 수 있도록 개선하면 좋을 것 같습니다.
- 여러 알고리즘 예를 들어 DQN과 같은 변형 알고리즘을 실험하여 기본 알고리즘과의 성능 차이를 비교해보면 좋을 것 같습니다.  
