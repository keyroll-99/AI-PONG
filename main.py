import json
import sys

import pygame

from ENV import PADDLE_WIDTH, WIDTH, HEIGHT, WHITE, BLACK
from NeuralNetwork import NeuralNetwork
from Paddle import Paddle, DataCollection
from Ball import Ball
from Old.NeuralNetwork import NeuralNetwork as OldNeuralNetwork

# Initialize Pygame
pygame.init()

# Font
font = pygame.font.Font(None, 36)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Create paddles and ball

p1_network = NeuralNetwork([6, 8, 8, 1], 0.01)
p1_network.from_json(json.load(open("data/p1_weights.json", "r")))

p2_network = NeuralNetwork([6, 8, 1], 0.01)

# initial_train_data = json.load(open("data/data_last.json", "r"))
# initial_train_data = list(filter(lambda x: x["player"]["target"] != 0.5, initial_train_data))
# p2_network.train(initial_train_data)
# p1_network.train(initial_train_data, 160_000)

data_collection = DataCollection()

p1_paddle = Paddle(PADDLE_WIDTH, HEIGHT // 2, p1_network, "p1", "p2")
p2_paddle = Paddle(WIDTH - PADDLE_WIDTH, HEIGHT // 2, p2_network, "p2", "p1")
ball = Ball()

# Create sprite groups
all_sprites = pygame.sprite.Group()
all_sprites.add(p1_paddle, p2_paddle, ball)

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
        p1_paddle.update_network(data_collection)
        p2_paddle.update_network(data_collection)
        ball.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update paddles and ball
    p1_paddle.move(ball, p2_paddle)#player_move()
    p2_paddle.move(ball, p1_paddle)
    ball.update()

    # Ball bouncing off paddles
    if pygame.sprite.collide_rect(ball, p1_paddle) or pygame.sprite.collide_rect(ball, p2_paddle):
        ball.speed_x = -ball.speed_x

    # Reset ball if it goes out of bounds
    if ball.rect.left <= 0:
        opponent_score += 1
        ball.reset()

    if ball.rect.right >= WIDTH:
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

    data_collection.collect(ball, p1_paddle, p2_paddle)

# Quit the game
pygame.quit()

# json.dump(data_collection.data, open("data/data.json", "w"))
# json.dump([layer for layer in p1_network.weights], open("data/player_weights.json", "w"))
# json.dump(p2_network.weights, open("data/opponent_weights.json", "w"))

print("Do you want to save network? [y/n]")

p1_weights = p1_network.to_json()
p2_weights = p2_network.to_json()
json.dump(p1_weights, open("data/p1_weights.json", "w"))
json.dump(p2_weights, open("data/p2_weights.json", "w"))


sys.exit()
