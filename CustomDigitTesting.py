import os
import torch
import torch.nn as nn
from Net import Net
import pygame
import numpy as np
import PIL.Image as Image
from scipy.ndimage import center_of_mass, shift # center mass recenters the drawing so its in the center

pygame.init()

window = (280, 280)
screen = pygame.display.set_mode(window)
pygame.display.set_caption("Draw a Digit | press 'S' to save and run model")

black = (0, 0, 0)
white = (255, 255, 255)

screen.fill(black)

mouse_down = False
last_position = None
brush_size = 10

def draw_circle(surface, position):
    pygame.draw.circle(surface, white, position, brush_size, 0)

#reload
def pre_process_image(image_path):
    #loading image
    pimg = Image.open(image_path).convert('L') #'L' makes this grayscale
    pimg = pimg.resize((28, 28), Image.Resampling.LANCZOS)
    #convert, normalizing (0 to 1 based on color)
    pimg = np.array(pimg).astype(np.float32) / 255.0

    y, x = center_of_mass(pimg)
    y = 14 - y
    x = 14 - x

    #shift to center and mode 'constant', cval=0 are making previous location black
    pimg = shift(pimg, (y, x), mode='constant', cval=0.0)

    #convert from 0 to 1 range to -1 to 1 because our model expects that
    pimg = (pimg - 0.5) / 0.5

    #mvoing from 28, 28 to what our model expects which is (1, 1, 28, 28)
    return torch.tensor(pimg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

folder_path = './models'
model_name = 'MNIST_model.pth'
file_path = os.path.join(folder_path, model_name)

model = Net()

state_dict = torch.load(file_path, weights_only=True)
model.load_state_dict(state_dict)
model.eval() # set to eval mode, don't worry about training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

running = True
while running:
    for event in pygame.event.get():
        #if user closes stop game
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            last_position = pygame.mouse.get_pos()
            mouse_down = True
        if event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

        if event.type == pygame.MOUSEMOTION:
            if mouse_down:
                mouse_position = pygame.mouse.get_pos()

                #clsoing any quick movements
                if last_position is not None:
                    pygame.draw.line(screen, white, last_position, mouse_position, brush_size*2)

                #draw our brush strokes
                draw_circle(screen, mouse_position)
                last_position = mouse_position

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                pygame.image.save(screen, 'digit.png')
                print('Saved image as digit.png')
                screen.fill(black)

                img_tensor = pre_process_image('digit.png')

                with torch.no_grad():
                    output = model(img_tensor.to(device))
                    _, predicted = torch.max(output, 1)

                    print(f"Number was: {predicted.item()}")

    pygame.display.flip()

pygame.quit()

