import pygame

from Ball import Ball
from ENV import PADDLE_WIDTH, PADDLE_HEIGHT, HEIGHT, WHITE, WIDTH
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

        if direction > 0.5:
            self.rect.y -= self.speed
        else:
            self.rect.y += self.speed

    def update_network(self, ball: Ball, opponent, is_your_win: bool):
        if not is_your_win:
            target = 0
            if self.rect.y > ball.rect.y:
                target = 1

            inputs = [
                ball.rect.x / WIDTH,
                ball.rect.y / HEIGHT,
                ball.speed_x / ball.ball_speed,
                ball.speed_y / ball.ball_speed,
                self.rect.y / HEIGHT,
                opponent.rect.y / HEIGHT,
            ]

            self.network.train(inputs, target)
