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

        # inicjalizujemy środowisko
        state = env.reset() # state to długości radarów np (2, 5, 6, 8, 2)
        total_reward = 0

        # każda klatka animacji
        for t in range(MAX_TRY): 

            # zapisz q_table do pliku
            env.handle_events(q_table)

            # na początku algorytm wykonuje losowe akcje, aby się nauczyć
            if mode == 1:
                if random.uniform(0, 1) < epsilon: # jeżeli epsilon jest większy od randomowego ułamka - wykonaj randomową akcję
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state]) # jeżeli inaczej to za pomocą arejki sprawdź jaka akcja jest najlepsza dla danego układu radarów
            else:
                action = np.argmax(q_table[state])

            # akcja -> wynik
            next_state, reward, done, _ = env.step(action) # wykonaj te akcję i przypisz jej wyniki do następnego state
            total_reward += reward # dodajemy reward danej klatki animacji do puli reward dla całego przejazdu

            # pobierz odpowiednia wartosc q z state, action pair
            q_value = q_table[state][action] # pobieram wartość wykonanej akcji
            best_q = np.max(q_table[next_state]) # maksymalna następna wartość akcji jaka będzie wykonanana następnie

            # równanie Bellmana
            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q) # zamiana wartości dla wykonanej akcji (by była mniej lub bardziej ważna przy następnej decyzji)
            
            # ustawienia dla następnej iteracji
            state = next_state

            # funkcja odpowiedzialna za narysowanie danej klatki
            env.render() 

            # na zakończenie próby, wyświetl wynik
            if done or t >= MAX_TRY - 1: 
                # To się dzieje jak się samochód wywali lub gdy próba trwa zbyt długo
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                print(epsilon)
                break

        if epsilon >= 0.005:            # po wykonaniu próby wartość epsilon się zmniejsza by przy decydowaniu czy samochód ma wziąć
            epsilon *= epsilon_decay    # randomową wartość czy wartość z tabelki coraz częściej brał wartość z tabelki



if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    MAX_EPISODES = 999999
    MAX_TRY = 200000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.2
    gamma = 0.75
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

    # dotępne tryby:
    # # #   1 - uczy się poprzez losowe ruchy (losowy ułamek)      2, 3, 4... inne - na podstawie zapisanej tablicy
    mode = 2  
    if mode == 1:
        q_table = np.zeros(num_box + (env.action_space.n,))  # pusta tablica q_table
    else:
        # q_table = np.zeros(num_box + (env.action_space.n,))  # dla pustej tablicy q_table
        q_table = load('./data/data.npy')  # dla zapisanej tablicy q_table

    simulate()
