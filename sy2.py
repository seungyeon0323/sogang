import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import random  # 랜덤 모듈을 사용하기 위한 import문입니다.


# plt.rcParams["font.family"] = "NanumGothic" 

plt.rcParams['font.family'] = 'AppleGothic.ttf'
plt.rcParams['axes.unicode_minus'] =False

np.set_printoptions(threshold=np.inf)


def display2D(array):
    for row in array:
        # 행 별로 반복합니다.
        for column in row:
            print(column.getSign(), end=" ")
        # getSign()은 방문하지 않은 원소를 나타내는 'O',
        # 클리프를 나타내는 'C', 방문한 원소를 나타내는 'X'를 반환합니다.
        print(end="\n")

def displayRewards(array):
    for row in array:
        # 열 별로 반복합니다.
        for column in row:
            print(column.getReward(), end=" ")
        print(end='\n')

def handleRewards(array, last_row, column):
    # handleRewards 함수는 보상을 설정하는 함수입니다.
    # 마지막 행의 일부 열에 대해 보상을 설정하며, 일반적으로 'C'인 구간을 클리프로 간주하여 보상을 부여합니다.

    # column - 2는 클리프 요소의 수에 해당합니다.
    for i in range(column - 2):
        # 클리프에 해당하는 열에 대해 보상을 설정합니다.
        array[last_row][i + 1].setReward(-100)
    # 마지막 열의 마지막 행에 대해 목표 지점의 보상을 설정할 수 있습니다.
    # array[last_row][column - 1].setReward(50)

def shiftAndAddArray(arrayToShift, newElement):
    # shiftAndAddArray 함수는 배열을 한 칸씩 이동하고 새로운 원소를 추가하는 함수입니다.
    # 이 함수는 주로 이동 평균을 계산하는 데 사용됩니다.

    for i in range(len(arrayToShift) - 1):
        # 배열을 한 칸씩 이동합니다.
        arrayToShift[i] = arrayToShift[i + 1]
    # 배열의 마지막 위치에 새로운 원소를 추가합니다.
    arrayToShift[len(arrayToShift) - 1] = newElement
    return arrayToShift

actionList = [0, 1, 2, 3]  # 0: 오른쪽, 1: 아래, 2: 위, 3: 왼쪽
# actionList는 가능한 행동을 나타내는 리스트입니다. 각 숫자는 특정 방향을 나타냅니다.
# 예를 들어, 0은 '오른쪽', 1은 '아래', 2는 '위', 3은 '왼쪽'에 해당합니다.


class States:
    """States 클래스는 그리드의 각 상태를 나타냅니다.

        Parameters:
        - sign: 현재 위치의 상태를 나타내는 문자 ('C': 클리프, 'S': 시작 지점, 'F': 종료 지점, 'O': 빈 공간, 'X': 에이전트 이동 경로)
        - reward: 현재 상태의 보상 값
        - row: 상태의 행 위치
        - column: 상태의 열 위치"""
    def __init__(self,sign,reward,row,column):
        self.sign = sign
        self.reward = reward
        self.row = row
        self.column = column

 
    def getReward(self):
        #현재 상태의 보상 값을 반환합니다.
        return self.reward

    def getSign(self):
        #현재 상태의 표지(상태)를 반환합니다.
        return self.sign

    def changeSign(self,newSign):
        #현재 상태의 표지(상태)를 변경합니다.
        self.sign = newSign


    def setReward(self,newReward):
        # 현재 상태의 보상 값을 변경합니다.
        self.reward = newReward

    def eGreedy(self,Qvalue,epsilon,last_row_index, last_column_index): 
        """
        epsilon-greedy 방식을 사용하여 행동을 선택합니다.

        Parameters:
        - Qvalue: 현재 Q-value 테이블
        - epsilon: epsilon 값
        - last_row_index: 그리드의 마지막 행 인덱스
        - last_column_index: 그리드의 마지막 열 인덱스

        Returns:
        - 선택된 행동 (인코딩된 정수 값)
        """
        newActionList = actionList.copy()  # 가능한 행동 중에서 선택할 수 있는 새로운 행동 목록을 만듭니다.

        # 특정 위치에서 금지된 행동을 제외합니다.
        if self.row == 0 and self.column == 0:  # 위쪽 또는 왼쪽으로 이동 불가능
            newActionList.remove(3)   # 왼쪽 제거
            newActionList.remove(2)   # 위쪽 제거
        elif self.row == last_row_index and self.column == 0:  # 아래쪽 또는 왼쪽으로 이동 불가능
            newActionList.remove(3)   # 왼쪽 제거
            newActionList.remove(1)   # 아래쪽 제거
        elif self.row == 0 and self.column == last_column_index:  # 위쪽 또는 오른쪽으로 이동 불가능
            newActionList.remove(0)   # 오른쪽 제거
            newActionList.remove(2)   # 위쪽 제거
        elif self.row == last_row_index and self.column == last_column_index:  # 아래쪽 또는 오른쪽으로 이동 불가능
            newActionList.remove(0)   # 오른쪽 제거
            newActionList.remove(1)   # 아래쪽 제거
        # 일반적인 경우에 대한 금지된 행동
        elif self.row == 0:  # 위쪽으로 이동 불가능
            newActionList.remove(2) # 위쪽 제거
        elif self.row == last_row_index:  # 아래쪽으로 이동 불가능
            newActionList.remove(1) # 아래쪽 제거
        elif self.column == 0:  # 왼쪽으로 이동 불가능
            newActionList.remove(3) # 왼쪽 제거
        elif self.column == last_column_index:  # 오른쪽으로 이동 불가능
            newActionList.remove(0) # 오른쪽 제거

        # epsilon-greedy 접근 방식을 사용하여 탐색과 활용을 보장합니다.
        probability = random.uniform(0, 1)  # 0과 1 사이의 랜덤한 숫자를 얻습니다.
        if probability < 1 - epsilon:  # 탐욕적인 행동 선택
            return self.findMaxIndex(Qvalue, newActionList, last_row_index, last_column_index)
        else:  # 랜덤한 행동 선택
            return random.choice(newActionList)

    """순수한 탐욕(Greedy) 행동을 반환합니다. 탐색 없이"""
    def ePureGreedy(self, Qvalue, last_row_index, last_column_index):
        newActionList = actionList.copy()  # 행동 목록 다시 복사
        if self.row == 0 and self.column == 0:  # 위 또는 왼쪽으로 이동 불가능
            newActionList.remove(3)  # 왼쪽 제거
            newActionList.remove(2)  # 위쪽 제거
        elif self.row == last_row_index and self.column == 0:  # 아래 또는 왼쪽으로 이동 불가능
            newActionList.remove(3)  # 왼쪽 제거
            newActionList.remove(1)  # 아래쪽 제거
        elif self.row == 0 and self.column == last_column_index:  # 위 또는 오른쪽으로 이동 불가능
            newActionList.remove(0)  # 오른쪽 제거
            newActionList.remove(2)  # 위쪽 제거
        elif self.row == last_row_index and self.column == last_column_index:  # 아래 또는 오른쪽으로 이동 불가능
            newActionList.remove(0)  # 오른쪽 제거
            newActionList.remove(1)  # 아래쪽 제거
        # 일반적인 경우
        elif self.row == 0:  # 위쪽으로 이동 불가능
            newActionList.remove(2)  # 위쪽 제거
        elif self.row == last_row_index:  # 아래쪽으로 이동 불가능
            newActionList.remove(1)  # 아래쪽 제거
        elif self.column == 0:  # 왼쪽으로 이동 불가능
            newActionList.remove(3)  # 왼쪽 제거
        elif self.column == last_column_index:  # 오른쪽으로 이동 불가능
            newActionList.remove(0)  # 오른쪽 제거

        return self.findMaxIndex(Qvalue, newActionList, last_row_index, last_column_index)


    """주어진 행동에 따라 상태를 변경합니다."""
    def takeAction(self, array, action, Qvalue, last_row_index, last_column_index, epsilon):
        # 주의: action은 정수 값입니다. 0이면 오른쪽, 1이면 아래쪽, 2이면 위쪽, 3이면 왼쪽

        # 행이 0인 경우, 위로 이동할 수 없음 - 유효한 동작을 위해 다시 eGreedy를 호출합니다.
        if self.row == 0 and action == 2:  # 행이 0이면 위로 이동할 수 없음 - 다시 eGreedy 호출
            newAction = self.eGreedy(Qvalue, epsilon, last_row_index, last_column_index)
            return self.takeAction(array, newAction, Qvalue, last_row_index, last_column_index, epsilon)
        # 행이 last_row_index인 경우, 아래로 이동할 수 없음
        elif self.row == last_row_index and action == 1:
            newAction = self.eGreedy(Qvalue, epsilon, last_row_index, last_column_index)
            return self.takeAction(array, newAction, Qvalue, last_row_index, last_column_index, epsilon)
        # 열이 last_column_index인 경우, 오른쪽으로 이동할 수 없음
        elif self.column == last_column_index and action == 0:
            newAction = self.eGreedy(Qvalue, epsilon, last_row_index, last_column_index)
            return self.takeAction(array, newAction, Qvalue, last_row_index, last_column_index, epsilon)
        # 열이 0인 경우, 왼쪽으로 이동할 수 없음
        elif self.column == 0 and action == 3:
            newAction = self.eGreedy(Qvalue, epsilon, last_row_index, last_column_index)
            return self.takeAction(array, newAction, Qvalue, last_row_index, last_column_index, epsilon)
        else:  # 모든 것이 유효한 경우
            if action == 0:  # 오른쪽
                next_state = array[self.row][self.column + 1]  # 상태 업데이트
            elif action == 1:  # 아래쪽
                next_state = array[self.row + 1][self.column]  # 상태 업데이트
            elif action == 2:  # 위쪽
                next_state = array[self.row - 1][self.column]  # 상태 업데이트
            elif action == 3:  # 왼쪽
                next_state = array[self.row][self.column - 1]
        return next_state


    """현재 상태에서 최대 Q-value를 갖는 행동의 인덱스를 반환합니다."""
    def findMaxIndex(self, Qvalue, actionList, last_row_index, last_column_index):
        maxIndex = np.argmax(Qvalue[self.row * 12 + self.column])
        return maxIndex



    def findMaxIndex(self, Qvalue, actionList, last_row_index, last_column_index):
        maxIndex = np.argmax(Qvalue[self.row * 12 + self.column])
        return maxIndex

    """현재 상태가 종료 상태인지 여부를 확인하여 에피소드를 종료합니다."""
    def isTerminal(self, last_row_index):
        # 현재 상태가 종료 상태(클리프 또는 도착 지점)이면 1 반환
        if self.row == last_row_index and self.column > 0:
            return 1
        else:
            return 0
  


# SARSA 알고리즘을 적용하는 함수
def applySarsa(array, row_number, column_number, episodes, steps, alpha, gamma, env):
    state_number = row_number * column_number
    last_row_index = row_number - 1
    last_column_index = column_number - 1
    action_number = 4
    epsilon_sarsa = 0.2  # 탐험을 조절하는 하이퍼파라미터
    # Q 테이블을 무작위로 초기화
    q_table = np.random.rand(state_number, action_number)
    # 종단 상태 초기화
    q_table[state_number - 1][0] = 0
    q_table[state_number - 1][1] = 0
    q_table[state_number - 1][2] = 0
    q_table[state_number - 1][3] = 0
    rewardArray = []  # 플로팅 목적으로 초기화
    episodeArray = []  # 플로팅 목적으로 초기화
    windowing_average_samples = 40  # 그래프를 부드럽게 만들기 위한 이동 평균 샘플 수
    averageRewardArray = np.zeros(windowing_average_samples)  # 초기화
    for i in range(episodes):  # 각 에피소드에 대한 루프
        totalReward = 0
        episodeArray.append(i)
        current_state = array[last_row_index][0]  # 초기 상태 S
        # Q에서 파생된 정책을 사용하여 상태 S에서 행동 A를 선택
        action = current_state.eGreedy(q_table, epsilon_sarsa, last_row_index, last_column_index) 
        count = 0
        while count in range(steps) and current_state.isTerminal(last_row_index) == 0:
            count = count + 1
            # 행동 A를 수행하고 보상 R, 다음 상태 S'를 관찰
            next_state = current_state.takeAction(array, action, q_table, last_row_index, last_column_index, epsilon_sarsa)
            reward_got = next_state.getReward()
            totalReward = totalReward + reward_got
            # Q에서 파생된 정책을 사용하여 상태 S'에서 행동 A'를 선택
            followingAction = next_state.eGreedy(q_table, epsilon_sarsa, last_row_index, last_column_index)
            q_table_next = q_table[((next_state.row) * column_number) + next_state.column][followingAction]
            q_table_current = q_table[((current_state.row) * column_number) + current_state.column][action]
            # Q 메서드의 업데이트
            q_table[((current_state.row) * column_number) + current_state.column][action] = q_table_current + alpha * (
                    reward_got + gamma * q_table_next - q_table_current)
            # 상태 업데이트
            current_state = next_state
            action = followingAction
        averageRewardArray = shiftAndAddArray(averageRewardArray, totalReward)
        valueToDraw = np.sum(averageRewardArray) / np.count_nonzero(averageRewardArray)
        rewardArray.append(valueToDraw)
    print("Sarsa TRAINING IS OVER")
    # 최종 경로를 그림
    drawPath(array, q_table, last_row_index, last_column_index, epsilon_sarsa)
    # 이동 평균을 망가뜨리는 세그먼트를 자름
    episodeArray = episodeArray[10:]
    rewardArray = rewardArray[10:]
    # 플로팅
    plt.plot(episodeArray, rewardArray)
    plt.xlabel('에피소드')
    plt.ylabel('총 보상(부드럽게 만들기)')
    print(averageRewardArray)


    """
    @params
        array = 구축한 상태 그리드
        row_number = 상태 그리드의 행 수
        column_number = 상태 그리드의 열 수
        episodes = 에피소드 수
        steps = 에피소드 내 단계 수
        alpha = 학습률
        gama = 할인 계수
    """

# Q-learning 알고리즘을 적용하는 함수
def applyQlearning(array, row_number, column_number, episodes, steps, alpha, gama, env):
    epsilon_q = 0.1
    state_number = row_number * column_number
    last_row_index = row_number - 1
    last_column_index = column_number - 1
    action_number = 4
    q_table = np.random.rand(state_number, action_number)
    # 종단 상태 초기화
    q_table[state_number - 1][0] = 0
    q_table[state_number - 1][1] = 0
    q_table[state_number - 1][2] = 0
    q_table[state_number - 1][3] = 0
    rewardArray = []
    episodeArray = []
    windowing_average_samples = 40
    averageRewardArray = np.zeros(windowing_average_samples)
    for i in range(episodes):  # 각 에피소드에 대한 루프
        step = 0
        current_state = array[3][0]  # 초기 상태 S
        count = 0
        totalReward = 0
        episodeArray.append(i)
        while count in range(steps) and current_state.isTerminal(last_row_index) == 0:
            step = step + 1
            count = count + 1
            # Q에서 파생된 정책을 사용하여 상태 S에서 행동 A를 선택
            action = current_state.eGreedy(q_table, epsilon_q, last_row_index, last_column_index)
            # 행동 A를 수행하고 보상 R, 다음 상태 S'를 관찰
            next_state = current_state.takeAction(array, action, q_table, last_row_index, last_column_index, epsilon_q)
            reward_got = next_state.getReward()
            totalReward = totalReward + reward_got
            actionMax = next_state.ePureGreedy(q_table, last_row_index, last_column_index)
            q_table_max = q_table[(next_state.row * column_number) + next_state.column][actionMax]
            q_table_current = q_table[(current_state.row * column_number) + current_state.column][action]
            q_table[(current_state.row * column_number) + current_state.column][action] = q_table_current + alpha * (
                    reward_got + (gama * q_table_max) - q_table_current)  # 업데이트
            current_state = next_state
        averageRewardArray = shiftAndAddArray(averageRewardArray, totalReward)
        valueToDraw = np.sum(averageRewardArray) / np.count_nonzero(averageRewardArray)
        rewardArray.append(valueToDraw)
    print("Q TRAINING IS OVER")
    episodeArray = episodeArray[10:]
    rewardArray = rewardArray[10:]
    drawPath(array, q_table, last_row_index, last_column_index, epsilon_q)
    plt.plot(episodeArray, rewardArray)
    plt.legend(['SARSA', 'Q learning'])
    print(averageRewardArray)
    plt.title(f'{windowing_average_samples}개의 연속된 에피소드를 평균화하여 부드럽게 만듦')
    # 전체적인 경계를 제한하여 시각화를 더욱 쾌적하게 만듦
    plt.ylim((-120, 0))
    plt.show()

# 최종 경로를 그리는 함수
def drawPath(array, Qvalue, last_row_index, last_column_index, epsilon):
    current_state = array[last_row_index][0]  # 초기 상태
    greedyAction = current_state.ePureGreedy(Qvalue, last_row_index, last_column_index)  # 행동 결정
    while current_state.isTerminal(last_row_index) == 0:
        next_state = current_state.takeAction(array, greedyAction, Qvalue, last_row_index, last_column_index, epsilon)
        next_state.changeSign("X")
        followingAction = next_state.ePureGreedy(Qvalue, last_row_index, last_column_index)
        current_state = next_state
        greedyAction = followingAction
    display2D(array)



def initialize_state_grid(row_number, column_number):
    stateGrid = [[States('O', 0, i, j) for j in range(column_number)] for i in range(row_number)]
    return stateGrid

def main():
    episode_number = 600
    steps = 500
    gamma = 0.9
    alpha = 0.25

    # Gym 환경 생성
    env = gym.make('CliffWalking-v0')

    # stateGrid 초기화 (2D 배열로 가정)
    row_number = env.observation_space.n
    column_number = env.observation_space.n  
    stateGrid = initialize_state_grid(row_number, column_number)

    display2D(stateGrid)
    print("OK HERE WE STOP")
    stateGrid[row_number - 1][0].changeSign('S')
    stateGrid[row_number - 1][column_number - 1].changeSign('F')
    for i in range(10):
        stateGrid[row_number - 1][i + 1].changeSign('C')

    display2D(stateGrid)
    handleRewards(stateGrid, row_number - 1, column_number)
    displayRewards(stateGrid)
    print("CONSTRUCTED")

    stateGrid2 = copy.deepcopy(stateGrid)

    applySarsa(stateGrid, row_number, column_number, episode_number, steps, alpha, gamma, env)
    applyQlearning(stateGrid2, row_number, column_number, episode_number, steps, alpha, gamma, env)

if __name__ == "__main__":
    main()




