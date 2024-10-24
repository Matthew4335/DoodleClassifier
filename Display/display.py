import pygame
import numpy as np
from PIL import Image

# Initialize Pygame
pygame.init()

# Set up the display
window_width, window_height = 1000, 620
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Doodle Classifier")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 70)
LIGHT_GRAY = (100, 100, 120)

# Set up drawing canvas
canvas_width, canvas_height = 560, 560
canvas_rect = pygame.Rect(10, 10, canvas_width, canvas_height)
canvas = pygame.Surface((canvas_width, canvas_height))
canvas.fill(BLACK)

# Example class names and probabilities (for demonstration purposes)
class_names = ["Helicopter", "Cruise Ship", "Tractor", "Cat", "House Plant",
               "Umbrella", "Octopus", "Windmill", "Bicycle", "Popsicle"]
probabilities = [98.19, 1.67, 0.07, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

# Function to capture the canvas and preprocess it for the neural network
def capture_canvas(canvas):
    canvas_array = pygame.surfarray.array3d(canvas)
    return canvas_array

# Preprocess the captured image
def preprocess_image(canvas_array):
    gray_image = np.mean(canvas_array, axis=2)
    image = Image.fromarray(gray_image).resize((28, 28))
    image = np.array(image) / 255.0
    return image.flatten().reshape(1, 784)

# Display the predictions and probabilities on the right side of the screen
def display_predictions(window, class_names, probabilities):
    font_large = pygame.font.SysFont(None, 40)
    font_small = pygame.font.SysFont(None, 35)

    # Display the highest probability class with a larger font
    top_class_text = font_large.render(f"{class_names[0]}", True, WHITE)
    window.blit(top_class_text, (canvas_width + 40, 40))

    # Display the other classes with smaller, grayed-out font
    for i in range(1, len(class_names)):
        class_text = font_small.render(f"{class_names[i]}", True, LIGHT_GRAY)
        window.blit(class_text, (canvas_width + 40, 40 + i * 50))

    top_percent_text = font_large.render(f"{probabilities[0]}%", True, WHITE)
    window.blit(top_percent_text, (canvas_width + 300, 40))

    # Display the other classes with smaller, grayed-out font
    for i in range(1, len(probabilities)):
        probabilities_text = font_small.render(f"{probabilities[i]}", True, LIGHT_GRAY)
        window.blit(probabilities_text, (canvas_width + 300, 40 + i * 50))


# Display instructions at the bottom
def display_instructions(window):
    font_instructions = pygame.font.SysFont(None, 30)
    instructions_text = "Left Click: Draw  |  C: Clear"
    instructions_surface = font_instructions.render(instructions_text, True, WHITE)
    window.blit(instructions_surface, (15, canvas_height + 25))


# Main loop
running = True
drawing = False

while running:
    window.fill(GRAY)
    window.blit(canvas, canvas_rect.topleft)  # Draw the canvas

    # Display the predictions and instructions
    display_predictions(window, class_names, probabilities)
    display_instructions(window)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle mouse events for drawing
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and canvas_rect.collidepoint(event.pos):  # Left click to draw
                drawing = True
            elif event.button == 3:  # Right click to erase
                drawing = False
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear canvas with 'C'
                canvas.fill(BLACK)

        if event.type == pygame.MOUSEMOTION and drawing:
            if canvas_rect.collidepoint(event.pos) and drawing:
                pygame.draw.circle(canvas, WHITE, (event.pos[0] - canvas_rect.x, event.pos[1] - canvas_rect.y), 10)

    pygame.display.flip()

pygame.quit()
