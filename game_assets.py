import numpy as np
import pygame
import random

MAP_SCALE = 30

PLAYER_SPEED = 4
PLAYER_SIZE = (30, 30)
FPS = 60
SPEED_CONVERSION = .5

SHEEP_SIZE = (40, 40)
SHEEP_MAX_SPEED = 3
SHEEP_SPEED = 1
NUMBER_SHEEP = 5

class GameMap:
    def __init__(self, filename):
        global MAP_WIDTH, MAP_HEIGHT
        self.map_file = filename
        self.boundaries = []
        self.pen_rects = []
        self.map_scale = MAP_SCALE
        self.sheep = []
        self.load_map()
        self.player = Player(self, (60, 60))

    def get_state(self):
        '''
        returns essential elements ready for input into a neural network
        player position
        sheep positions
        location of goal
        is finished
        '''
        player_position = self.player.position

        sheep_positions = []
        for sheep in self.sheep:
            sheep_positions.append(sheep.position)

        sheep_positions = np.array(sheep_positions)
        is_finished = 1 if self.game_over() == True else 0
        positions = []
        positions.append(player_position)
        positions.extend(sheep_positions)
        return np.array(positions), is_finished

    def set_state(self, state):
        '''
        state: [:, 2] Rows of (x, y) pairs
        The first position is assumed to be player position, followed by sheep positions
        '''
        expected_length = len(self.sheep) + 1
        if len(state) != expected_length:
            print("Tried to set state with size of %s, %s expected. " % (len(state), expected_length))
            return
        self.player.position = state[0, :]
        # Assign values to all of the sheep
        for shp_num, sheep in enumerate(self.sheep):
            sheep.position = state[shp_num+1, :]

    def advance_moves(self, action=1, distance=1):
        '''
        Advance player and sheep moves for this game cycle
        action: [1-5] Player's move
           1: Stay still
           2: Move North
           3: Move East
           4. Move South
           5. Move West
        distance: Number of times to move the distance specified in the agent's speed
        '''
        UP = action == 2
        DOWN = action == 4
        LEFT = action == 5
        RIGHT = action == 3
        for _ in range(distance):
            self.player.move_player(UP, DOWN, LEFT, RIGHT)
            for sheep in self.sheep:
                sheep.advance_move()

    def load_map(self):
        self.map = None
        combined = []
        with open(self.map_file) as map_file:
            self.map_shape = [int(x) for x in map_file.readline().split()]
            for line in map_file:
                this_line = [str(s) for s in line.strip()]
                combined.append(this_line)
        self.map = np.array(combined)
        self.set_w_h()
        self.set_player()
        self.generate_boundaries()
        self.generate_goal_areas()
        self.load_sheep()

    def set_player(self):
        PLAYER_CHAR = "D"
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if self.map[j, i] == PLAYER_CHAR:
                    self.player = Player(self, (i*self.map_scale, j*self.map_scale))

    def load_sheep(self):
        sheep_locs = []
        SHEEP_CHAR = "S"
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if self.map[j, i] == SHEEP_CHAR:
                    sheep_locs.append((i*self.map_scale, j*self.map_scale))
        self.sheep = self.generate_sheep(sheep_locs)


    def generate_boundaries(self):
        BOUNDARY_CHAR = '#'
        self.boundaries = []
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if self.map[j, i] == BOUNDARY_CHAR:
                    self.boundaries.append(pygame.Rect(i*self.map_scale, j*self.map_scale, self.map_scale, self.map_scale))

    def generate_goal_areas(self):
        PEN_CHAR = 'P'
        self.pen_rects = []
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                if self.map[j, i] == PEN_CHAR:
                    self.pen_rects.append(pygame.Rect(i*self.map_scale, j*self.map_scale, self.map_scale, self.map_scale))

    def game_over(self):
        all_sheep_in_pen = None
        for shp in self.sheep:
            shp_rect = pygame.Rect(shp.position, shp.size)
            if shp_rect.collidelist(self.pen_rects) == -1:
                all_sheep_in_pen = False
                break
            else:
                all_sheep_in_pen = True
        return all_sheep_in_pen

    def get_score(self, in_pen_score=25, dis_from_pen_penalty=.1):
        score = 0
        missing_sheep = []
        # Add sheep in pen
        for shp in self.sheep:
            shp_rect = pygame.Rect(shp.position, shp.size)
            if shp_rect.collidelist(self.pen_rects) == -1:
                missing_sheep.append(shp)
            else:
                score += in_pen_score
        # Measure aprox distance from pen
        if dis_from_pen_penalty:
            for sheep in missing_sheep:
                score -= self.dist_nearest_pen(sheep) * dis_from_pen_penalty
        return score

    def dist_nearest_pen(self, sheep):
        lowest_dis = 1000000
        for pen in self.pen_rects:
            distance = np.linalg.norm(sheep.position - pen[0:2])
            if distance < lowest_dis:
                lowest_dis = distance
        return lowest_dis

    def set_w_h(self):
        self.width = self.map_scale * self.map_shape[0]
        self.height = self.map_scale * self.map_shape[1]

    def generate_sheep(self, sheep_loc=None):
        global NUMBER_SHEEP
        if sheep_loc is None:
            sheep_loc = []
            while len(sheep_loc) < NUMBER_SHEEP:
                random_pos = self.random_position()
                if not self.collides_boundary(random_pos, (20, 20)):
                    sheep_loc.append(random_pos)

        sheep = []
        for loc in sheep_loc:
            sheep.append(Sheep(self, loc))
        return sheep

    def random_position(self):
        random_x = random.random() * self.width // 1
        random_y = random.random() * self.height // 1
        return (random_x, random_y)

    def allowed_position(self, original_position, new_position, width_height):
        '''
        Checks through the boundary boxes, if a collision with the new_position is found
        then change new_position to be next to that block instead.
        return: a valid position
        '''

        if not self.collides_boundary(new_position, width_height):
            return new_position
        if self.collides_boundary(original_position, width_height):
            return new_position

        corrected_x_pos = self.find_non_collisioin(original_position, new_position, width_height, (1,0))
        corrected_y_pos = self.find_non_collisioin(corrected_x_pos, new_position, width_height, (0, 1))

        return corrected_y_pos

    def find_non_collisioin(self, original_position, position, width_height, isolated_move_dir):
        '''
        Takes a position that's causing a collision, and backtracks to a non-colliding place
        isolated_move_dir: either (0, 1) or (1, 0)
        '''
        move_vector = position - original_position
        isolated_move = np.multiply(move_vector, isolated_move_dir)
        test_pos = original_position + isolated_move

        if not self.collides_boundary(test_pos, width_height):
            return test_pos

        # Find boxes that are collided with
        collided_with = []
        for boundary in self.boundaries:
            if self.collides_boundary(test_pos, width_height, [boundary]):
                collided_with.append(boundary)

        # Find the most constraining block to compare with
        result = None
        for boundary in collided_with:
            if not result:
                result = boundary
                continue
            if isolated_move[0]:
                if isolated_move[0] > 0:
                    if boundary[0] < result[0]:
                        result = boundary
                else:
                    if boundary[0] + boundary[2] > result[0] + result[2]:
                        result = boundary
            else:
                if isolated_move[1] > 0:
                    if boundary[1] < result[1]:
                        result = boundary
                else:
                    if boundary[1] + boundary[3] > result[1] + result[3]:
                        result = boundary

        # Move the position so that it is touching the most constraining block
        compare = np.dot(original_position, isolated_move_dir)
        if isolated_move.sum() > 0:
            compare = compare + np.dot(isolated_move_dir, width_height)

        bound_compare = np.dot(result[:2], isolated_move_dir)
        if isolated_move.sum() < 0:
            bound_compare = bound_compare + np.dot(isolated_move_dir, np.array(result[2:4]))

        new_position = original_position + abs(compare - bound_compare) * np.array(isolated_move_dir)
        if self.collides_boundary(new_position, width_height):
            new_position = original_position
        return new_position


    def collides_boundary(self, position, width_height, boundaries=None):
        if boundaries is None:
            boundaries = self.boundaries
        this_rect = pygame.Rect(position[0], position[1], width_height[0], width_height[1])
        return not this_rect.collidelist(boundaries) == -1



class Sheep:
    def __init__(self, map, position):
        '''
        position: (2d array)
        velocity: (2d array)
        max_speed: (float)
        acceleration: (float)
        '''
        global SHEEP_SPEED, SHEEP_MAX_SPEED, SPEED_CONVERSION, SHEEP_SIZE
        self.map = map
        self.timestill = 0
        self.AWARENESS_DIS = 80
        self.size = SHEEP_SIZE

        self.direction = (0, 0)
        self.position = np.array(position)
        self.previous_position = self.position
        self.destination = self.position
        self.max_speed = SHEEP_MAX_SPEED * SPEED_CONVERSION
        self.normal_speed = SHEEP_SPEED * SPEED_CONVERSION


    def advance_move(self):
        if self.at_destination() or self.player_nearby() or self.at_previous_location():
            self.timestill += 1
            self.destination = self.move_location()
        if not self.at_destination():
            self.direction = self.destination - self.position
            self.direction = self.direction / np.linalg.norm(self.direction)
            speed = self.max_speed if self.player_nearby() else self.normal_speed
            move_dir = self.destination - self.position
            move_magnitude = np.linalg.norm(move_dir)
            if move_magnitude > .001:
                percent_to_move = min(move_magnitude, speed) / move_magnitude
                proposed_position = self.position + percent_to_move * move_dir
                self.update_position(proposed_position)

    def at_destination(self, destination=None, tolerance=.1):
        if destination is None:
            destination = self.destination
        return np.allclose(self.position, self.destination, rtol=tolerance)

    def at_previous_location(self, update=True, tolerance=.001):
        '''
        If update is True, it is assumed this function is only called once per move cycle
        '''
        result = np.allclose(self.previous_position, self.position, rtol=tolerance)
        return result

    def move_location(self):
        direction = np.array((0, 0))
        if self.player_nearby():
            direction = self.away_from_player()
        else:
            direction = self.random_move()
        MAX_DISTANCE = 100
        result = self.position + direction * random.random() * MAX_DISTANCE * self.herd_awareness() / self.AWARENESS_DIS
        return result

    def player_nearby(self, awareness_dis=None):
        if awareness_dis is None:
            awareness_dis = min(self.AWARENESS_DIS, self.herd_awareness())
        player_pos = self.map.player.position
        distance_to_player = np.linalg.norm((self.position - player_pos))
        return distance_to_player < awareness_dis

    def herd_awareness(self):
        return max(self.AWARENESS_DIS - self.herd_effect(), self.AWARENESS_DIS // 2)

    def herd_effect(self):
        '''
        For each sheep in the near vicinity lower this sheeps perceptions by one
        '''
        amount = 0
        for sheep in self.map.sheep:
            sheep_pos = sheep.position
            if np.allclose(sheep_pos, self.position, .00001):
                continue
            if np.linalg.norm(self.position - sheep_pos) < self.AWARENESS_DIS:
                amount += 1
        return amount

    def away_from_player(self):
        player_pos = self.map.player.position
        direction = self.position - player_pos
        return direction / np.linalg.norm(direction)

    def random_move(self):
        '''
        stay still for 1 seconds, then become more likely to move as it gets closer to 10 seconds
        '''
        global FPS
        direction = np.array((0, 0))
        will_move = ((self.timestill - 1 * FPS) / (10 * FPS) ) > random.random()
        if will_move:
            direction = self.random_dir()
        return direction

    def update_position(self, proposed_position):
        self.previous_position = np.array(self.position)
        self.position = self.map.allowed_position(self.position, proposed_position, self.size)
        if not self.at_destination(proposed_position) and not self.at_previous_location():
            self.timestill = 0

    def random_dir(self):
        x = random.choice([1, 0, -1])
        y = random.choice([1, 0, -1])
        dir = np.array((x, y))

        dir = dir / max(np.linalg.norm(dir), .0001)
        return dir


class Player:
    def __init__(self, map, position, manually_controlled=True):
        '''
        position: (2d numpy array)
        '''
        global PLAYER_SPEED, PLAYER_ACCELERATION, PLAYER_SIZE, SPEED_CONVERSION
        self.map = map
        self.position = position
        self.speed = PLAYER_SPEED * SPEED_CONVERSION
        self.manually_controlled = manually_controlled
        self.size = PLAYER_SIZE
        self.direction = (0, 1)

    def move_direction(self):
        return np.array((0, 0))

    def check_for_input(self, pressed):
        up = bool(pressed[pygame.K_UP])
        down = bool(pressed[pygame.K_DOWN])
        left = bool(pressed[pygame.K_LEFT])
        right = bool(pressed[pygame.K_RIGHT])
        self.move_player(up, down, left, right)

    def move_player(self, UP=False, DOWN=False, LEFT=False, RIGHT=False):
        '''
        Converts cardinal directions to a vector direction, then moves player in that direction
        '''
        dir = np.array((0, 0))
        if UP:
            dir[1] = -1
        elif DOWN:
            dir[1] = 1
        if LEFT:
            dir[0] = -1
        elif RIGHT:
            dir[0] = 1
        self.move_direction(dir)

    def move_direction(self, dir):
        dir_mag = np.linalg.norm(dir)
        dir = dir / max(dir_mag, .0001)
        proposed_position = self.position + self.speed * dir
        if dir_mag > .0001:
            self.direction = dir
        self.position = self.map.allowed_position(self.position, proposed_position, self.size)

def initiate_game(filename, graphics=True):
    gm = GameMap("test.txt")
    gm.run_game(graphics)

if __name__ == "__main__":

    print(gm.map)
