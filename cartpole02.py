import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 입력층에서 첫 번째 은닉층으로의 연결
        self.fc2 = nn.Linear(128, 128)  # 첫 번째 은닉층에서 두 번째 은닉층으로의 연결
        self.fc3 = nn.Linear(128, 2)  # 두 번째 은닉층에서 출력층으로의 연결, CartPole 환경의 행동 공간은 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 마지막은 활성화 함수를 사용하지 않음
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        rv = random.random()
        if rv < epsilon:
            return random.randint(0, 1)  # 무작위로 행동 선택
        else:
            return out.argmax().item()  # 네트워크의 출력 중 가장 큰 값의 인덱스를 선택

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=50000)  # 50000개 이상부터는 예전 과거꺼는 삭제하면서 돌아감

    def size(self):
        return len(self.buffer)
    
    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)

        s_list, a_list, r_list, s_prime_list, done_mask_list = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            done_mask_list.append([done_mask])

        return (
            torch.tensor(s_list, dtype=torch.float),
            torch.tensor(a_list),
            torch.tensor(r_list),
            torch.tensor(s_prime_list, dtype=torch.float),
            torch.tensor(done_mask_list)
        )
    
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime, done_mask = memory.sample(32)
        
        q_out = q(s)
        q_a = q.out.gather(1,a)

        max.q_prime = q.target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        loss_ = F.mse_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    
    # DQN 모델 생성
    q = DQN()
    q_target = DQN()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameter(), lr=0.0005)

    socre = 0.0
    for n_epi in range(3000):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))

        s, _ = env.reset()
        done = False
        while not done: #terminate상태가 되었는지 
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                s_price, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                done_mask =0.0 if done else 1.0 #done_mask 0.0 끝 1.0 안끝남

                memory.put((s,a,r,s_price, done_mask))
                s = s_price

                score += r
                if done: break

            if memory.size() > 2000:
                train(q, q_target, memory, optimizer)

            if (n_epi != 0) and (n_epi % 20 ==0):
                q_target.load_state_dict(q.state_dict())

                print("n_epoo:(), score:{:.1f}, n_buffer:{}, eps:{:.1f}.formate(n_epi, score/20, memory.size(), epsilon*100) ")
                score = 0.0

if __name__ == '__main__':
    main()
