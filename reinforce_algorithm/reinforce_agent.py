""" 이 코드는 '파이썬과 케라스로 배우는 강화학습' 책 안의 코드를 pytorch코드로 바꾸어본 코드입니다!!
    코드동작환경 : Mac os / pytorch 1.0.0(no GPU) / Pycharm """

"""readme: 기존 케라스로 작성된 코드에서는 Cross entropy loss function을 클래스 안에 정의하고 구
           동되는 방식이나 파이토치로 작성한 이 코드의 경우는 프레임워크의 차이가 있기 때문에 train_model 
           부분 내 정의하여 주었습니다. 참고바랍니다. """

import copy
import pylab
import numpy as np
from environment import Env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

EPISODES = 2500

# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99 
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.torch.load('./save_model/reinforce_trained.h5')
    
    # 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성(정책신경망 정의부분)
    def build_model(self):

        input = nn.Linear(self.state_size, 24, bias=True)
        activation_in = nn.ReLU()
        hidden1 = nn.Linear(24, 24, bias=True)
        activation1 = nn.ReLU()
        hidden2 = nn.Linear(24, 24, bias=True)
        activation2 = nn.ReLU()
        output = nn.Linear(24, self.action_size, bias=True)
        activation_out = nn.Softmax()

        model = nn.Sequential(input, activation_in, hidden1, activation1, hidden2, activation2, output, activation_out)

        return model


    # 정책신경망으로 행동 선택
    def get_action(self, state):

        state = Variable(torch.from_numpy(state)).float()
        policy = self.model(state[0])

        return np.random.choice(self.action_size, 1, p=policy.detach().numpy())[0]
    
    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0].tolist())
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act.tolist())

    # 정책신경망 업데이트
    def train_model(self):

        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        states = Variable(torch.FloatTensor(np.array(self.states)))
        actions = Variable(torch.FloatTensor(np.array(self.actions)))
        discounted_rewards = Variable(torch.FloatTensor(discounted_rewards))

        pre = self.model(states)

        # 크로스 엔트로피 오류함수 정의 및 계산
        action_prob = torch.sum(actions * pre, 1)
        cross_entropy = torch.log(action_prob) * discounted_rewards
        cost = -torch.sum(cross_entropy)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #한 에피소드 동안의 state, action, 개별 reward를 모았던 리스트 초기화
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트의 생성
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 정책신경망 업데이트
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            torch.save(agent.model.state_dict(), "./save_model/reinforce.h5")

