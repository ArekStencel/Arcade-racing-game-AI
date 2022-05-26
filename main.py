import sys
import numpy as np
from numpy import load
import math
import random

import gym
import gym_game

def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset() #state to długości radarów np (2,5,6,8,2)
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY): #każda klatak animacji

            # save current q_table to file
            env.handle_events(q_table)

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon: #jeżeli epsilon jest większy od randomowego ułamka - wykonaj randomową akcję
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state]) #jeżeli inaczej to za pomocą arejki sprawdź jaka akcja jest najlepsza dla danego układu radarów
            # Do action and get result
            next_state, reward, done, _ = env.step(action) #wykonaj te akcję i przypisz jej wyniki do następnego state
            total_reward += reward #dodajemy reward danej klatki animacji do puli reward dla całego przejazdu

            # Get correspond q value from state, action pair
            q_value = q_table[state][action] #pobieram wartość wykonanej akcji
            best_q = np.max(q_table[next_state]) #maxymalna następna wartość akcji jaka będzie wykonanana następnie

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q) #zamiana wartości dla wykonanej akcji (by była mniej lub bardziej ważna przy następnej decyzji
            # Set up for the next iteration
            state = next_state #następuje następny state

            # Draw games
            env.render() #funkcja odpowiedzialna za narysowanie danej klatki

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1: #To się dzieje jak się samochód wywali lub gdy próba trwa zbyt długo
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                print(epsilon)
                break

        # exploring rate decay
        if epsilon >= 0.005:            #po wykonaniu próby wartość epsilon się zmniejsza by przy decydowaniu czy samochód ma wziąć
            epsilon *= epsilon_decay    # randomową wartość czy wartość z tableki coraz częściej brał wartość z tabelki



if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    MAX_EPISODES = 99999
    MAX_TRY = 20000
    epsilon = 1
    epsilon_decay = 0.99
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

    #q_table = load('data.npy')
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()
