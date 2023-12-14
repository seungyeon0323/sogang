![image](https://github.com/seungyeon0323/sogang/assets/153755354/bacbf388-af8b-49fb-a7c6-38da086685ac)# cliff walking
  강화학습 기초 프로젝트

## 💻 프로젝트 소개
* Cliff Walking 문제는 강화 학습 분야에서 제가 적용할 수 있을것이라고 판단하여 선택하게 되었습니다. 
이 프로젝트를 통해, 강화 학습 알고리즘의 기본 개념을 실제로 적용해보고, 그 효과와 한계를 직접 실험하며 이해하고자하였습니다.

### ⏰ 프로젝트 기간
* 2023-10-31 ~ 2023-12-14

### 💿 개발환경
- gym.make('CliffWalking-v0’)
- python 
- Numpy, maplotlib, seaborn, gymnasium

### 🔖하이퍼파라미터
- env: gym 라이브러리의 CliffWalking 환경을 사용
- episodes: 실행할 총 에피소드의 수(500)
- learn_rate: 학습률로, Q 값을 얼마나 빠르게 업데이트할지 (0.5)discount: 할인율로, 미래 보상을 현재 가치로 얼마나 할인할지 (0.9)
- explore_rate: 탐색률로, 에이전트가 무작위 행동을 선택할지 (1.0)
  
## 📌 주요기능
### Q - learning
- q_vals: Q-테이블을 초기화. 
- rewards_sum: 각 에피소드별 총 보상을 저장할 리스트를 초기화.
- 에피소드 반복:
</br>for ep in range(episodes): 지정된 에피소드 수만큼 반복을 실행
- 에이전트의 행동:
</br>while not done: 에피소드가 끝날 때까지 (done이 True가 될 때까지) 반복
- 현재 상태 계산:
</br>state, _ = calculate_state(position, q_vals): 현재 에이전트의 위치(position)를 기반으로 상태를 계산
</br>action = select_action(state, q_vals, explore_rate): 현재 상태에 기반하여 행동을 선택 이때 explore_rate에 따라 무작위 행동이 선택될 수도 있음
</br>new_pos = move_agent(position, action): 선택된 행동을 기반으로 에이전트의 위치를 업데이트
</br>next_state, max_next_q = calculate_state(new_pos, q_vals): 새 위치에서의 다음 상태와 그 상태의 최대 Q 값을 계산
</br>reward, done = determine_reward(next_state): 새로운 상태에 대한 보상을 받고, 에피소드가 종료되었는지 여부를 결정
</br>total_reward += reward: 총 보상에 현재 보상을 합산
</br>q_vals = adjust_q_values(q_vals, state, action, reward, max_next_q, discount, learn_rate): 벨만 방정식을 사용하여 Q-테이블을 업데이트
</br>position = new_pos: 에이전트의 위치를 새 위치로 업데이트 
- ε-탐욕적 정책 (Epsilon-Greedy Policy): 에이전트가 탐험(새로운 행동 시도)과 이용(현재 정보를 기반으로 최적의 행동 선택) 사이에서 균형을 맞추도록 도움

### sarsa
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


## 📖프로젝트 후 느낀점
- Q-Learning과 SARSA 알고리즘의 학습 과정과 성능을 시각화함으로써 두 알고리즘이 어떻게 다른지, 어떤 상황에서 한 알고리즘이 다른 알고리즘보다 나은 성능을 보이는지를 이해할 수 있습니다
 시각화를 통하여 알고리즘의 안정성을 확인할 수 있었고 직관적으로 분석이 가능하다는 것을 느꼈습니다.
- Q-Learning이 더 빠르게 최적의 경로를 찾아가는 경향이 있는 반면, SARSA는 현재 정책에 더 의존하면서 조심스럽게 탐색한다는 것을 직접 실습을 통해 알 수 있었습니다..


## 📚프로젝트 개선점
- 탐색률을 고정된 값이 아닌 에피소드가 진행됨에 따라 점점 감소하는 방식으로 설정하여 초기에는 탐색을 더 많이 하고, 시간이 지남에 따라 최적의 정책에 수렴하도록 할 수 있도록 개선하면 좋을 것 같습니다.
- 여러 알고리즘 예를 들어 DQN과 같은 변형 알고리즘을 실험하여 기본 알고리즘과의 성능 차이를 비교해보면 좋을 것 같습니다.  
