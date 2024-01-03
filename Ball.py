import pygame
import random

from ENV import BALL_SIZE, WIDTH, HEIGHT, WHITE, BALL_SPEED


class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_SIZE, BALL_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.ball_speed = BALL_SPEED
        self.speed_x = self.ball_speed * random.choice([1, -1])
        self.speed_y = self.ball_speed * random.choice([1, -1])

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Ball bouncing off walls
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y = -self.speed_y

    def reset(self):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed_x = self.ball_speed * random.choice([1, -1])
        self.speed_y = self.ball_speed * random.choice([1, -1])
