import sys

import pygame

from ENV import PADDLE_WIDTH, WIDTH, HEIGHT, WHITE, BLACK
from NeuralNetwork import NeuralNetwork
from Paddle import Paddle, DataCollection
from Ball import Ball

# Initialize Pygame
pygame.init()

# Font
font = pygame.font.Font(None, 36)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Create paddles and ball

player_network = NeuralNetwork(1, [8])
opponent_network = NeuralNetwork(1, [8])
data_collection = DataCollection()

player_paddle = Paddle(PADDLE_WIDTH, HEIGHT // 2, player_network)
opponent_paddle = Paddle(WIDTH - PADDLE_WIDTH, HEIGHT // 2, opponent_network)
ball = Ball()

# Create sprite groups
all_sprites = pygame.sprite.Group()
all_sprites.add(player_paddle, opponent_paddle, ball)

# Score variables
player_score = 0
opponent_score = 0

# Game loop
clock = pygame.time.Clock()
running = True


while running:
    if player_score == 3 or opponent_score == 3:
        player_score = 0
        opponent_score = 0
        # player_paddle.update_network(data_collection, True)
        # opponent_paddle.update_network(data_collection, False)
        ball.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update paddles and ball
    player_paddle.move(ball, opponent_paddle)
    opponent_paddle.move(ball, player_paddle)
    ball.update()

    # Ball bouncing off paddles
    if pygame.sprite.collide_rect(ball, player_paddle) or pygame.sprite.collide_rect(ball, opponent_paddle):
        ball.speed_x = -ball.speed_x

    # Reset ball if it goes out of bounds
    if ball.rect.left <= 0:
        opponent_score += 1
        player_paddle.update_network(data_collection, True)
        opponent_paddle.update_network(data_collection, False)
        data_collection.data.clear()
        ball.reset()

    if ball.rect.right >= WIDTH:
        player_paddle.update_network(data_collection, True)
        opponent_paddle.update_network(data_collection, False)
        data_collection.data.clear()
        player_score += 1
        ball.reset()

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    all_sprites.draw(screen)

    # Draw scores
    player_text = font.render(str(player_score), True, WHITE)
    opponent_text = font.render(str(opponent_score), True, WHITE)
    screen.blit(player_text, (WIDTH // 4, 20))
    screen.blit(opponent_text, (3 * WIDTH // 4 - opponent_text.get_width(), 20))

    # Update display
    pygame.display.flip()

    # Set the frame rate
    clock.tick(60)

    data_collection.collect(ball, player_paddle, opponent_paddle)

# Quit the game
pygame.quit()
sys.exit()
