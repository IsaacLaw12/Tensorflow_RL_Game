import pygame
import time
import math
import numpy as np
from assets import GameMap, PLAYER_SIZE, SHEEP_SIZE

gamemap = GameMap("map.txt")
x_bounds = gamemap.width
y_bounds = gamemap.height

end_font = None
dog_img = pygame.image.load('dog.png')
p_size = [int(x*1.5) for x in PLAYER_SIZE]
dog_img = pygame.transform.scale(dog_img, p_size)
sheep_img = pygame.image.load('sheep.png')
s_size = [int(x*1.5) for x in SHEEP_SIZE]
sheep_img = pygame.transform.scale(sheep_img, s_size)


def run_game():
    pygame.init()
    global end_font
    end_font = pygame.font.SysFont(None, 40)
    screen = pygame.display.set_mode((x_bounds, y_bounds))
    running = True

    while running:
        gamemap.load_map()
        start_time = time.time()
        quit = run_game_instance(screen)
        end_time = time.time()
        score = end_time - start_time
        if quit:
            break
        if not play_again(screen, score):
            running = False

def play_again(screen, score):
    play_again = None
    while play_again is None:
        render_end_screen(screen, score)
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_RETURN]:
            gamemap.load_map()
            play_again = True
        if pressed[pygame.K_q]:
            play_again = False
        if check_for_quit():
            play_again = False
    return play_again

def render_end_screen(screen, score):
    draw_background(screen)
    end_text = end_font.render("You finished in %.1f seconds" % score, True, (255, 255, 255))
    continue_text  = end_font.render("Press 'q' to quit or hit 'enter' to play again", True, (255, 255, 255))
    print
    screen.blit(end_text, (x_bounds //2 - end_text.get_width() // 2, y_bounds // 2 - end_text.get_height() - 40))
    screen.blit(continue_text, (x_bounds//2 - continue_text.get_width() // 2, y_bounds // 2 - continue_text.get_height() ))


    pygame.display.flip()


def run_game_instance(screen):
    clock = pygame.time.Clock()
    while not gamemap.game_over():
        if check_for_quit():
            return True
        read_input()
        render_graphics(screen)
        clock.tick(60)
    return False

def read_input():
    pressed = pygame.key.get_pressed()
    gamemap.player.check_for_input(pressed)
    for sheep in gamemap.sheep:
        sheep.advance_move()

def check_bounds(position):
    result = []
    result = [min(x) for x in zip(position, [x_bounds - 30, y_bounds - 30])]
    result = [max(x) for x in zip(result, [10, 10])]
    return result

def render_graphics(screen):
    draw_background(screen)
    draw_goals(screen)
    draw_barriers(screen)
    draw_sheep(screen)
    draw_player(screen)
    pygame.display.flip()

def draw_background(screen):
    screen.fill((80, 128, 80), pygame.Rect(0, 0, x_bounds, y_bounds))

def draw_barriers(screen):
    for bound_rect in gamemap.boundaries:
        screen.fill( (30, 100, 30), bound_rect)

def draw_goals(screen):
    for pen_rect in gamemap.pen_rects:
        screen.fill( (255, 186, 115), pen_rect)

def draw_sheep(screen):
    for sheep in gamemap.sheep:
        xpos, ypos = sheep.position
        direction = sheep.direction
        r_sheep_img = rotated_image(sheep_img, direction)
        screen.blit(r_sheep_img, (xpos,ypos))

def draw_player(screen):
    xpos, ypos = gamemap.player.position
    r_dog_img = rotated_image(dog_img, gamemap.player.direction)
    screen.blit(r_dog_img, (xpos, ypos))

def rotated_image(surface_obj, direction):
    '''
    Assumes images default to direction (0, 1)
    returns a surface of the image rotated to the specified direction
    direction: unit length vector
    '''
    angle = math.acos(np.dot((0, 1), direction))
    if direction [0] < 0:
        angle = angle * -1
    angle = angle * 180 / math.pi
    result = pygame.transform.rotate(surface_obj, angle)
    return result

def check_for_quit():
    done = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
                done = True
    return done

if __name__ == "__main__":
    run_game()
