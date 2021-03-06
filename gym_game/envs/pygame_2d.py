import pygame
import math
from numpy import save

import settings
from settings import *

screen_width = 1700
screen_height = 900
check_point = ((1135, 725), (1415, 515), (1465, 275), (970, 315), (970,300), (970,170), (840, 135), (710, 315), (265, 205), (240, 515), (505, 775))

class Car:
    def __init__(self, car_file, map_file, pos):
        self.surface = pygame.image.load(car_file)
        self.map = pygame.image.load(map_file)
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = pos
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.goal = False
        self.check_flag = False
        self.distance = 0
        self.time_spent = 0
        self.draw_radars = settings.draw_radars
        self.draw_hitboxes = settings.draw_hitboxes
        self.draw_checkpoints = settings.draw_checkpoints
        self.draw_rewards = settings.draw_rewards

        for d in range(-90, 95, 45):
            self.check_radar(d)

        for d in range(-90, 95, 45):
            self.check_radar_for_draw(d)

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)

    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (0, 0, 0) , (x, y), 5)

    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self):
        self.is_alive = True
        for p in self.four_points:
            if self.map.get_at((int(p[0]), int(p[1]))) == (34, 32, 52, 255):
                self.is_alive = False
                break

    def check_radar(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (34, 32, 52, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])


    def check_radar_for_draw(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (34, 32, 52, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])

    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 70:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
            else:
                self.goal = False

        self.cur_distance = dist

    def update(self):
        # sprawd?? pr??dko????
        self.speed -= 0.5
        if self.speed > 10:
            self.speed = 10
        if self.speed < 1:
            self.speed = 1

        # sprawd?? pozycje
        self.rotate_surface = rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # oblicz wszystkie 4 punkty kolizyjne
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 51]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

class PyGame2D:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.car = Car('./assets/car.png', './assets/map.png', [650, 725])
        self.font = pygame.font.SysFont("Arial", 30)
        self.game_speed = 60
        self.total_reward = 0

    def action(self, action):
        if action == 0:
            self.car.speed += 2
        if action == 1:
            self.car.angle += 5
        elif action == 2:
            self.car.angle -= 5

        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, 95, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0

        if self.car.check_flag:
            self.car.check_flag = False
            reward = 1000 - self.car.time_spent
            self.car.time_spent = 0

        if not self.car.is_alive:
            reward = -10000 + self.car.distance

        elif self.car.goal:
            reward = 10000

        self.total_reward += reward
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    # zwraca state'a
    def observe(self):
        
        radars = self.car.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)

        return tuple(ret)

    # rysuj gre
    def view(self):
        self.screen.blit(self.car.map, (0, 0))

        self.car.radars_for_draw.clear()
        for d in range(-90, 95, 45):
            self.car.check_radar_for_draw(d)

        # sprawdzamy czy mamy wyrenderowa?? konkretny element
        if self.car.draw_checkpoints:
            pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        if self.car.draw_hitboxes:
            self.car.draw_collision(self.screen)
        if self.car.draw_radars:
            self.car.draw_radar(self.screen)
        self.car.draw(self.screen)

        # czy pokaza?? wynik
        if self.car.draw_rewards:
            text = self.font.render(str(self.total_reward), True, (255, 255, 0))
            text_rect = text.get_rect()
            text_rect.center = (50, 50)
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)

    # skr??ty klawiszowe odpowiedzialne za w????czenie/wy????czenie
    # radar??w, checkpoint??w, zapisu, hitbox??w, punkt??w
    def handle_events(self, content):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save('./data/data.npy', content)
                    print("data saved to file")
                elif event.key == pygame.K_r:
                    print("radars view toggled")
                    self.car.draw_radars = self.car.draw_radars ^ 1
                    settings.draw_radars = self.car.draw_radars
                elif event.key == pygame.K_t:
                    print("hitboxes view toggled")
                    self.car.draw_hitboxes = self.car.draw_hitboxes ^ 1
                    settings.draw_hitboxes = self.car.draw_hitboxes
                elif event.key == pygame.K_y:
                    print("checkpoints view toggled")
                    self.car.draw_checkpoints = self.car.draw_checkpoints ^ 1
                    settings.draw_checkpoints = self.car.draw_checkpoints
                elif event.key == pygame.K_u:
                    print("reward view toggled")
                    self.car.draw_rewards = self.car.draw_rewards ^ 1
                    settings.draw_rewards = self.car.draw_rewards


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
