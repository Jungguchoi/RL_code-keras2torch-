""" 이 코드는 '파이썬과 케라스로 배우는 강화학습' 책 안의 코드를 Pytorch코드로 바꾸어본 코드입니다!!
    코드동작환경 : Mac os / pytorch 1.0.0(no GPU) / Pycharm """

import copy
import pylab
import random
import numpy as np
from environment import Env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


EPISODES = 1000


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.torch.load('./save_model/deep_sarsa_trained.h5')


    # 상태가 입력 큐함수가 출력인 인공신경망 생성(모델 정의부분)
    def build_model(self):
        hidden1 = nn.Linear(self.state_size, 30, bias=True)
        activation1 = nn.ReLU()
        hidden2 = nn.Linear(30, 30, bias=True)
        activation2 = nn.ReLU()
        hidden3 = nn.Linear(30, 30, bias=True)
        activation3 = nn.ReLU()
        output = nn.Linear(30, 5, bias=True)

        model = nn.Sequential(hidden1, activation1, hidden2, activation2, hidden3, activation3, output)

        return model

    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return random.randrange(self.action_size)
        else:
            # 모델로부터 행동 산출
            state = np.float32(state)
            state = Variable(torch.from_numpy(state))
            q_values = self.model(state)
            max_index = torch.argmax(q_values)

            return max_index

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)

        state = Variable(torch.from_numpy(state))
        target = self.model(state)[0]

        # 살사의 큐함수 업데이트 식
        if done:
            target[action] = reward
        else:
            next_state = Variable(torch.from_numpy(next_state))
            target[action] = (reward + self.discount_factor *
                              self.model(next_state)[0][next_action])

        # 출력 값 reshape
        target = target.view(1, 5)

        # 인공신경망 업데이트
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        pre = self.model(state)
        cost = loss(pre, target)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSARSAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])


        while not done:
            # env 초기화
            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)


            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_sarsa_.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            torch.save(agent.model.state_dict(), "./save_model/deep_sarsa.h5")


