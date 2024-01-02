import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle constants
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 100

# Ball constants
BALL_SIZE = 20

# Neural Network constants
INPUT_SIZE = 6  # Ball x, ball y, ball speed x, ball speed y, player paddle y, opponent paddle y
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 8
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.weights_input_hidden1 = [
            [random.uniform(-1, 1) for _ in range(INPUT_SIZE)]
            for _ in range(HIDDEN_SIZE_1)
        ]
        self.biases_hidden1 = [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE_1)]

        self.weights_hidden1_hidden2 = [
            [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE_1)]
            for _ in range(HIDDEN_SIZE_2)
        ]
        self.biases_hidden2 = [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE_2)]

        self.weights_hidden2_output = [random.uniform(-1, 1) for _ in range(HIDDEN_SIZE_2)]
        self.bias_output = random.uniform(-1, 1)

    def predict(self, inputs):
        # Forward pass
        hidden1_layer = [
            sigmoid(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.weights_input_hidden1, self.biases_hidden1)
        ]

        hidden2_layer = [
            sigmoid(sum(w * h1 for w, h1 in zip(weights, hidden1_layer)) + bias)
            for weights, bias in zip(self.weights_hidden1_hidden2, self.biases_hidden2)
        ]

        output = sigmoid(sum(w * h2 for w, h2 in zip(self.weights_hidden2_output, hidden2_layer)) + self.bias_output)
        return output

    def train(self, inputs, target):
        # Forward pass
        hidden1_layer = [
            sigmoid(sum(w * inp for w, inp in zip(weights, inputs)) + bias)
            for weights, bias in zip(self.weights_input_hidden1, self.biases_hidden1)
        ]

        hidden2_layer = [
            sigmoid(sum(w * h1 for w, h1 in zip(weights, hidden1_layer)) + bias)
            for weights, bias in zip(self.weights_hidden1_hidden2, self.biases_hidden2)
        ]

        output = sigmoid(sum(w * h2 for w, h2 in zip(self.weights_hidden2_output, hidden2_layer)) + self.bias_output)

        # Calculate errors
        output_error = target - output
        hidden2_errors = [
            output_error * self.weights_hidden2_output[i] * hidden2_layer[i] * (1 - hidden2_layer[i])
            for i in range(HIDDEN_SIZE_2)
        ]
        hidden1_errors = [
            sum(hidden2_errors[j] * self.weights_hidden1_hidden2[i][j] * hidden1_layer[i] * (1 - hidden1_layer[i])
                for j in range(HIDDEN_SIZE_2))
            for i in range(HIDDEN_SIZE_1)
        ]

        # Update weights and biases
        self.weights_hidden2_output = [w + LEARNING_RATE * output_error * hidden2_layer[i] for i, w in enumerate(self.weights_hidden2_output)]
        self.bias_output += LEARNING_RATE * output_error

        for i in range(HIDDEN_SIZE_2):
            self.weights_hidden1_hidden2[i] = [
                w + LEARNING_RATE * hidden2_errors[i] * hidden1_layer[j]
                for j, w in enumerate(self.weights_hidden1_hidden2[i])
            ]
            self.biases_hidden2[i] += LEARNING_RATE * hidden2_errors[i]

        for i in range(HIDDEN_SIZE_1):
            self.weights_input_hidden1[i] = [
                w + LEARNING_RATE * hidden1_errors[i] * inputs[j]
                for j, w in enumerate(self.weights_input_hidden1[i])
            ]
            self.biases_hidden1[i] += LEARNING_RATE * hidden1_errors[i]

# Create a neural network for the opponent paddle
opponent_network = NeuralNetwork()

class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def move(self, speed):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and self.rect.top > 0:
            self.rect.y -= speed
        if keys[pygame.K_DOWN] and self.rect.bottom < HEIGHT:
            self.rect.y += speed

    def move_ai(self, ball):
        # AI controlled by the neural network
        inputs = [
            ball.rect.x / WIDTH,   # Ball x position
            ball.rect.y / HEIGHT,  # Ball y position
            ball.speed_x / 5,      # Ball x speed (normalized)
            ball.speed_y / 5,      # Ball y speed (normalized)
            self.rect.y / HEIGHT,  # Player paddle y position (normalized)
            opponent_paddle.rect.y / HEIGHT  # Opponent paddle y position (normalized)
        ]

        # Make a decision using the neural network
        decision = opponent_network.predict(inputs)

        # Move the paddle based on the decision
        if decision > 0.5:
            self.rect.y -= 5
        else:
            self.rect.y += 5

# ... rest of the code remains the same ...

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update paddles and ball
    player_paddle.move(8)
    opponent_paddle.move_ai(ball)
    ball.update()

    # Ball bouncing off paddles
    if pygame.sprite.collide_rect(ball, player_paddle) or pygame.sprite.collide_rect(ball, opponent_paddle):
        ball.speed_x = -ball.speed_x

    # Reset ball if it goes out of bounds
    if ball.rect.left <= 0:
        opponent_score += 1

        # Autolearning after losing a point
        inputs = [
            ball.rect.x / WIDTH,   # Ball x position
            ball.rect.y / HEIGHT,  # Ball y position
            ball.speed_x / 5,      # Ball x speed (normalized)
            ball.speed_y / 5,      # Ball y speed (normalized)
            player_paddle.rect.y / HEIGHT,  # Player paddle y position (normalized)
            opponent_paddle.rect.y / HEIGHT  # Opponent paddle y position (normalized)
        ]
        opponent_network.train(inputs, 1)  # Training to move up

        ball.reset()

    if ball.rect.right >= WIDTH:
        player_score += 1
        ball.reset()

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    all_sprites.draw(screen)

    # Draw scores
    font = pygame.font.Font(None, 36)
    player_text = font.render(str(player_score), True, WHITE)
    opponent_text = font.render(str(opponent_score), True, WHITE)
    screen.blit(player_text, (WIDTH // 4, 20))
    screen.blit(opponent_text, (3 * WIDTH // 4 - opponent_text.get_width(), 20))

    # Update display
    pygame.display.flip()

    # Set the frame rate
    clock.tick(60)

# Quit the game
pygame.quit()
sys.exit()
