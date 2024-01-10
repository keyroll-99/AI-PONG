import pygame

from Ball import Ball
from ENV import PADDLE_WIDTH, PADDLE_HEIGHT, HEIGHT, WHITE, WIDTH, BALL_SPEED
from NeuralNetwork import NeuralNetwork


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, network: NeuralNetwork, name: str, opponent_name):
        super().__init__()
        self.epoch = 0
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = 8
        self.network = network
        self.name = name
        self.opponent_name = opponent_name

    def player_move(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] and self.rect.top > 0:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN] and self.rect.bottom < HEIGHT:
            self.rect.y += self.speed

    def move(self, ball: Ball, opponent):

        inputs = [
            ball.rect.x / WIDTH,
            ball.rect.y / HEIGHT,
            ball.speed_x / ball.ball_speed,
            ball.speed_y / ball.ball_speed,
            self.rect.y / HEIGHT,
            opponent.rect.y / HEIGHT,
        ]

        direction = self.network.predict(inputs)[0][0] # [0] result [1] bias

        if direction > 0.6 and self.rect.top > 0:
            self.rect.y -= self.speed
        elif direction < 0.4 and self.rect.bottom < HEIGHT:
            self.rect.y += self.speed

    def update_network(self, data):
        learning_data_length = len(data.data)
        learning_data_length = int(learning_data_length * 0.7)
        learning_data = data.data[:learning_data_length]
        self.network.train(learning_data)


def avg(lst):
    return sum(lst) / len(lst)


def get_target(player_rect: pygame.rect, ball_rect: pygame.rect):
    target = 1
    if player_rect.y < ball_rect.y:
            target = 0

    return target


class DataCollection:
    data = []

    def collect(self, ball: Ball, p1: Paddle, p2: Paddle):
        self.data.append({
            "ball": {
                "x": ball.rect.x,
                "y": ball.rect.y,
                "speed_x": ball.speed_x,
                "speed_y": ball.speed_y
            },
            "player": {
                "x": p1.rect.x,
                "y": p1.rect.y,
                "target": get_target(p1.rect, ball.rect)
            },
            "opponent": {
                "x": p2.rect.x,
                "y": p2.rect.y,
            }
        })
