import pygame

from Ball import Ball
from ENV import PADDLE_WIDTH, PADDLE_HEIGHT, HEIGHT, WHITE, WIDTH, BALL_SPEED
from NeuralNetwork import NeuralNetwork


class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, network: NeuralNetwork):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speed = 8
        self.network = network

    def move(self, ball: Ball, opponent):

        inputs = [
            ball.rect.x / WIDTH,
            ball.rect.y / HEIGHT,
            ball.speed_x / ball.ball_speed,
            ball.speed_y / ball.ball_speed,
            self.rect.y / HEIGHT,
            opponent.rect.y / HEIGHT,
        ]

        direction = self.network.predict(inputs)

        if direction > 0.5 and self.rect.top > 0:
            self.rect.y -= self.speed
        elif direction < 0.5 and self.rect.bottom < HEIGHT:
            self.rect.y += self.speed

    def update_network(self, data, is_player=True):
        for item in data.data:
            ball = item["ball"]
            player = item["player"]
            opponent = item["opponent"]

            inputs = [
                ball["x"] / WIDTH,
                ball["y"] / HEIGHT,
                ball["speed_x"] / BALL_SPEED,
                ball["speed_y"] / BALL_SPEED,
                player["y"] / HEIGHT,
                opponent["y"] / HEIGHT,
            ]

            self.network.train(inputs, player["target"] if is_player else opponent["target"])


class DataCollection:
    data = []

    def collect(self, ball: Ball, player: Paddle, opponent: Paddle):
        self.data.append({
            "ball": {
                "x": ball.rect.x,
                "y": ball.rect.y,
                "speed_x": ball.speed_x,
                "speed_y": ball.speed_y
            },
            "player": {
                "x": player.rect.x,
                "y": player.rect.y,
                "target": 1 if player.rect.y < ball.rect.y else 0
            },
            "opponent": {
                "x": opponent.rect.x,
                "y": opponent.rect.y,
                "target": 1 if opponent.rect.y < ball.rect.y else 0
            }
        })
