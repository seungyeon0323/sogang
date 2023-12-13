import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 입력층에서 첫 번째 은닉층으로의 연결, 입력 차원: 4, 출력 차원: 128
        self.fc2 = nn.Linear(128, 128)  # 첫 번째 은닉층에서 두 번째 은닉층으로의 연결, 입력 차원: 128, 출력 차원: 128
        self.fc3 = nn.Linear(128, 2)  # 두 번째 은닉층에서 출력층으로의 연결, 출력 차원: 2 (CartPole 환경의 행동 공간)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU 활성화 함수를 적용한 첫 번째 은닉층
        x = F.relu(self.fc2(x))  # ReLU 활성화 함수를 적용한 두 번째 은닉층
        x = self.fc3(x)  # 마지막은 활성화 함수를 사용하지 않음
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)  # DQN 모델에 입력을 전달하여 출력을 얻음
        rv = random.random()
        if rv < epsilon:
            return random.randint(0, 1)  # epsilon-greedy 전략을 사용하여 무작위 또는 네트워크의 예측 중 하나를 선택
        else:
            return out.argmax().item()  # 네트워크의 출력 중 가장 큰 값의 인덱스를 선택

# Experience Replay를 위한 ReplayBuffer 클래스 정의
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=50000)  # 50000개 이상부터는 예전 데이터를 삭제하면서 사용

    def size(self):
        return len(self.buffer)
    
    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # 버퍼에서 무작위로 미니배치 추출

        # 미니배치에서 상태, 행동, 보상, 다음 상태, 종료 여부의 리스트를 초기화합니다.
        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []

        # 미니배치의 각 transition에서 정보를 추출하여 각 리스트에 추가합니다.
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition  # transition에서 상태, 행동, 보상, 다음 상태, 종료 여부를 추출
            s_list.append(s)  # 현재 상태를 리스트에 추가
            a_list.append([a])  # 행동을 리스트에 추가 (행동을 하나의 리스트로 감싸서 추가)
            r_list.append([r])  # 보상을 리스트에 추가 (보상을 하나의 리스트로 감싸서 추가)
            s_prime_list.append(s_prime)  # 다음 상태를 리스트에 추가
            done_mask_list.append([done_mask])  # 종료 여부를 리스트에 추가 (종료 여부를 하나의 리스트로 감싸서 추가)

        return (
            # 미니배치에서 추출한 정보들을 PyTorch 텐서로 변환
            # - 상태 (`s_list` 및 `s_prime_list`)는 dtype을 torch.float로 지정하여 부동소수점 형태의 텐서로 변환
            # - 행동 (`a_list`), 보상 (`r_list`), 종료 여부 (`done_mask_list`)는 추가적인 변환 없이 텐서로 변환
            
            #현재 상태의 텐서를 생성
            torch.tensor(s_list, dtype=torch.float),
            
            # 행동, 보상 종료 여부의 텐서를 생성
            torch.tensor(a_list),
            torch.tensor(r_list),
            torch.tensor(s_prime_list, dtype=torch.float),
            
            # 종료 여부의 경우 dtype을 지정하지 않아도 됨
            torch.tensor(done_mask_list)
        )

# DQN 모델 학습 함수 정의
def train(q, q_target, memory, optimizer, gamma=0.98):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(32)  # 미니배치 크기 32로 샘플링
        
        q_out = q(s)  # DQN 모델에 현재 상태를 입력으로 전달하여 Q값을 얻음
        q_a = q_out.gather(1, a)  # 선택된 행동에 대한 Q값 추출

        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # 타겟 네트워크를 이용하여 최대 Q값 계산
        target = r + gamma * max_q_prime * done_mask  # 타겟 값 계산

        loss = F.mse_loss(q_a, target)  # MSE 손실 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 메인 함수 정의
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    
    # DQN 모델 생성
    q = DQN()
    q_target = DQN()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=0.0005)  # Adam 옵티마이저 사용

    score = 0.0
    for n_epi in range(3000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # 탐험 확률 감소

        s, _ = env.reset()
        done = False
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)  # epsilon-greedy 전략을 사용하여 행동 샘플링
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0  # done_mask: 0.0이면 종료, 1.0이면 미완료

            memory.push((s, a, r, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)  # 학습 수행

        if (n_epi != 0) and (n_epi % 20 == 0):
            q_target.load_state_dict(q.state_dict())  # 타겟 네트워크 업데이트

            print("n_epi: {}, score: {:.1f}, n_buffer: {}, eps: {:.1f}".format(n_epi, score / 20, memory.size(), epsilon * 100))
            score = 0.0

if __name__ == '__main__':
    main()

            