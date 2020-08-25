import sys
from copy import deepcopy

"""
laser_tank.py

This file contains a class representing Laser Tank game map. You should make use of this class in your solver.

Running this file directly with a valid testcase file given as an argument launches an interactive instance of the
Laser Tank game.

COMP3702 2020 Assignment 1 Support Code

Last updated by njc 16/08/20
"""


class LaserTankMap:
    """
    Instance of a LaserTank game map. You may use this class and its functions
    directly or duplicate and modify it in your solution. To ensure Tester
    functions correctly, you should avoid modifying this file directly.
    """

    # input file symbols
    LAND_SYMBOL = ' '
    WATER_SYMBOL = 'W'
    OBSTACLE_SYMBOL = '#'
    BRIDGE_SYMBOL = 'B'
    BRICK_SYMBOL = 'K'
    ICE_SYMBOL = 'I'
    TELEPORT_SYMBOL = 'T'
    FLAG_SYMBOL = 'F'

    MIRROR_UL_SYMBOL = '1'
    MIRROR_UR_SYMBOL = '2'
    MIRROR_DL_SYMBOL = '3'
    MIRROR_DR_SYMBOL = '4'

    PLAYER_UP_SYMBOL = '^'  # note: player always starts on a land tile
    PLAYER_DOWN_SYMBOL = 'v'
    PLAYER_LEFT_SYMBOL = '<'
    PLAYER_RIGHT_SYMBOL = '>'

    ANTI_TANK_UP_SYMBOL = 'U'
    ANTI_TANK_DOWN_SYMBOL = 'D'
    ANTI_TANK_LEFT_SYMBOL = 'L'
    ANTI_TANK_RIGHT_SYMBOL = 'R'
    ANTI_TANK_DESTROYED_SYMBOL = 'X'

    VALID_SYMBOLS = [LAND_SYMBOL, WATER_SYMBOL, OBSTACLE_SYMBOL, BRIDGE_SYMBOL, BRICK_SYMBOL, ICE_SYMBOL,
                     TELEPORT_SYMBOL, FLAG_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_UR_SYMBOL, MIRROR_DL_SYMBOL,
                     MIRROR_DR_SYMBOL, PLAYER_UP_SYMBOL, PLAYER_DOWN_SYMBOL, PLAYER_LEFT_SYMBOL, PLAYER_RIGHT_SYMBOL,
                     ANTI_TANK_UP_SYMBOL, ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL,
                     ANTI_TANK_DESTROYED_SYMBOL]

    # move symbols (i.e. output file symbols)
    MOVE_FORWARD = 'f'
    TURN_LEFT = 'l'
    TURN_RIGHT = 'r'
    SHOOT_LASER = 's'
    MOVES = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, SHOOT_LASER]

    # directions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # move return statuses
    SUCCESS = 0
    COLLISION = 1
    GAME_OVER = 2

    # render characters
    MAP_GLYPH_TABLE = {LAND_SYMBOL: '   ', WATER_SYMBOL: 'WWW', OBSTACLE_SYMBOL: 'XXX', BRIDGE_SYMBOL: '[B]',
                       BRICK_SYMBOL: '[K]', ICE_SYMBOL: '-I-', TELEPORT_SYMBOL: '(T)', FLAG_SYMBOL: ' F ',
                       MIRROR_UL_SYMBOL: ' /|', MIRROR_UR_SYMBOL: '|\\ ', MIRROR_DL_SYMBOL: ' \\|',
                       MIRROR_DR_SYMBOL: '|/ ', ANTI_TANK_UP_SYMBOL: '[U]', ANTI_TANK_DOWN_SYMBOL: '[D]',
                       ANTI_TANK_LEFT_SYMBOL: '[L]', ANTI_TANK_RIGHT_SYMBOL: '[R]', ANTI_TANK_DESTROYED_SYMBOL: '[X]'}
    PLAYER_GLYPH_TABLE = {UP: '[^]', DOWN: '[v]', LEFT: '[<]', RIGHT: '[>]'}

    # symbols which are movable for each direction
    MOVABLE_SYMBOLS = {UP: [BRIDGE_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_UR_SYMBOL, ANTI_TANK_UP_SYMBOL,
                            ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       DOWN: [BRIDGE_SYMBOL, MIRROR_DL_SYMBOL, MIRROR_DR_SYMBOL, ANTI_TANK_DOWN_SYMBOL,
                              ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       LEFT: [BRIDGE_SYMBOL, MIRROR_UL_SYMBOL, MIRROR_DL_SYMBOL, ANTI_TANK_UP_SYMBOL,
                              ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_LEFT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL],
                       RIGHT: [BRIDGE_SYMBOL, MIRROR_UR_SYMBOL, MIRROR_DR_SYMBOL, ANTI_TANK_UP_SYMBOL,
                               ANTI_TANK_DOWN_SYMBOL, ANTI_TANK_RIGHT_SYMBOL, ANTI_TANK_DESTROYED_SYMBOL]
                       }

    def __init__(self, x_size, y_size, grid_data, player_x=None, player_y=None, player_heading=None):
        """
        Build a LaserTank map instance from the given grid data.
        :param x_size: width of map
        :param y_size: height of map
        :param grid_data: matrix with dimensions (y_size, x_size) where each element is a valid symbol
        """
        self.x_size = x_size
        self.y_size = y_size
        self.grid_data = grid_data

        # extract player position and heading if none given
        if player_x is None and player_y is None and player_heading is None:
            found = False
            for i in range(y_size):
                row = self.grid_data[i]
                for j in range(x_size):
                    if row[j] == self.PLAYER_UP_SYMBOL or row[j] == self.PLAYER_DOWN_SYMBOL or \
                            row[j] == self.PLAYER_LEFT_SYMBOL or row[j] == self.PLAYER_RIGHT_SYMBOL:
                        found = True
                        self.player_x = j
                        self.player_y = i
                        self.player_heading = {self.PLAYER_UP_SYMBOL: self.UP,
                                               self.PLAYER_DOWN_SYMBOL: self.DOWN,
                                               self.PLAYER_LEFT_SYMBOL: self.LEFT,
                                               self.PLAYER_RIGHT_SYMBOL: self.RIGHT}[row[j]]
                        # replace the player symbol with land tile
                        row[j] = self.LAND_SYMBOL
                        break
                if found:
                    break
            if not found:
                raise Exception("LaserTank Map Error: Grid data does not contain player symbol")
        elif player_x is None or player_y is None or player_heading is None:
            raise Exception("LaserTank Map Error: Incomplete player coordinates given")
        else:
            self.player_x = player_x
            self.player_y = player_y
            self.player_heading = player_heading

    @staticmethod
    def process_input_file(filename):
        """
        Process the given input file and create a new map instance based on the input file.
        :param filename: name of input file
        """
        f = open(filename, 'r')

        rows = []
        i = 0
        for line in f:
            # skip optimal steps and time limit
            if i > 1 and len(line.strip()) > 0:
                rows.append(list(line.strip()))
            i += 1

        f.close()

        row_len = len(rows[0])
        for row in rows:
            assert len(row) == row_len, "LaserTank Map Error: Mismatch in row length"

        num_rows = len(rows)

        tp_count = 0
        player_count = 0
        flag_count = 0
        for row in rows:
            for symbol in row:
                if symbol == LaserTankMap.TELEPORT_SYMBOL:
                    tp_count += 1
                elif symbol == LaserTankMap.PLAYER_UP_SYMBOL or \
                        symbol == LaserTankMap.PLAYER_DOWN_SYMBOL or \
                        symbol == LaserTankMap.PLAYER_LEFT_SYMBOL or \
                        symbol == LaserTankMap.PLAYER_RIGHT_SYMBOL:
                    player_count += 1
                elif symbol == LaserTankMap.FLAG_SYMBOL:
                    flag_count += 1
                elif symbol not in LaserTankMap.VALID_SYMBOLS:
                    raise Exception("LaserTank Map Error: Invalid symbol in input file")
        assert tp_count % 2 == 0, "LaserTank Map Error: Unmatched teleport symbol"
        assert tp_count < 3, "LaserTank Map Error: Too many teleport symbols"
        assert player_count > 0, "LaserTank Map Error: No initial player position given"
        assert player_count < 2, "LaserTank Map Error: More than one initial player position given"
        assert flag_count > 0, "LaserTank Map Error: No goal position given"
        assert flag_count < 2, "LaserTank Map Error: More than one goal position given"

        return LaserTankMap(row_len, num_rows, rows)

    def apply_move(self, move):
        """
        Apply a player move to the map.
        :param move: self.MOVE_FORWARD, self.TURN_LEFT, self.TURN_RIGHT or self.SHOOT_LASER
        :return: LaserTankMap.SUCCESS if move was successful and no collision (or no effect move) occurred,
                 LaserTankMap.COLLISION if the move resulted collision or had no effect,
                 LaserTankMap.GAME_OVER if the move resulted in game over
        """

        if move == self.MOVE_FORWARD:
            # get coordinates for next cell
            if self.player_heading == self.UP:
                next_y = self.player_y - 1
                next_x = self.player_x
                if next_y < 0:
                    return self.COLLISION
            elif self.player_heading == self.DOWN:
                next_y = self.player_y + 1
                next_x = self.player_x
                if next_y >= self.y_size:
                    return self.COLLISION
            elif self.player_heading == self.LEFT:
                next_y = self.player_y
                next_x = self.player_x - 1
                if next_x < 0:
                    return self.COLLISION
            else:
                next_y = self.player_y
                next_x = self.player_x + 1
                if next_x >= self.x_size:
                    return self.COLLISION

            # handle special tile types
            if self.grid_data[next_y][next_x] == self.ICE_SYMBOL:
                # handle ice tile - slide until first non-ice tile or blocked
                if self.player_heading == self.UP:
                    for i in range(next_y, -1, -1):
                        if self.grid_data[i][next_x] != self.ICE_SYMBOL:
                            if self.grid_data[i][next_x] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(i, next_x):
                                # if blocked, stop on last ice cell
                                next_y = i + 1
                                break
                            else:
                                next_y = i
                                break
                elif self.player_heading == self.DOWN:
                    for i in range(next_y, self.y_size):
                        if self.grid_data[i][next_x] != self.ICE_SYMBOL:
                            if self.grid_data[i][next_x] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(i, next_x):
                                # if blocked, stop on last ice cell
                                next_y = i - 1
                                break
                            else:
                                next_y = i
                                break
                elif self.player_heading == self.LEFT:
                    for i in range(next_x, -1, -1):
                        if self.grid_data[next_y][i] != self.ICE_SYMBOL:
                            if self.grid_data[next_y][i] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(next_y, i):
                                # if blocked, stop on last ice cell
                                next_x = i + 1
                                break
                            else:
                                next_x = i
                                break
                else:
                    for i in range(next_x, self.x_size):
                        if self.grid_data[next_y][i] != self.ICE_SYMBOL:
                            if self.grid_data[next_y][i] == self.WATER_SYMBOL:
                                # slide into water - game over
                                return self.GAME_OVER
                            elif self.cell_is_blocked(next_y, i):
                                # if blocked, stop on last ice cell
                                next_x = i - 1
                                break
                            else:
                                next_x = i
                                break
            if self.grid_data[next_y][next_x] == self.TELEPORT_SYMBOL:
                # handle teleport - find the other teleporter
                tpy, tpx = (None, None)
                for i in range(self.y_size):
                    for j in range(self.x_size):
                        if self.grid_data[i][j] == self.TELEPORT_SYMBOL and i != next_y and j != next_x:
                            tpy, tpx = (i, j)
                            break
                    if tpy is not None:
                        break
                if tpy is None:
                    raise Exception("LaserTank Map Error: Unmatched teleport symbol")
                next_y, next_x = (tpy, tpx)
            else:
                # if not ice or teleport, perform collision check
                if self.cell_is_blocked(next_y, next_x):
                    return self.COLLISION

            # check for game over conditions
            if self.cell_is_game_over(next_y, next_x):
                return self.GAME_OVER

            # no collision and no game over - update player position
            self.player_y = next_y
            self.player_x = next_x

        elif move == self.TURN_LEFT:
            # no collision or game over possible
            if self.player_heading == self.UP:
                self.player_heading = self.LEFT
            elif self.player_heading == self.DOWN:
                self.player_heading = self.RIGHT
            elif self.player_heading == self.LEFT:
                self.player_heading = self.DOWN
            else:
                self.player_heading = self.UP

        elif move == self.TURN_RIGHT:
            # no collision or game over possible
            if self.player_heading == self.UP:
                self.player_heading = self.RIGHT
            elif self.player_heading == self.DOWN:
                self.player_heading = self.LEFT
            elif self.player_heading == self.LEFT:
                self.player_heading = self.UP
            else:
                self.player_heading = self.DOWN

        elif move == self.SHOOT_LASER:
            # set laser direction
            if self.player_heading == self.UP:
                heading = self.UP
                dy, dx = (-1, 0)
            elif self.player_heading == self.DOWN:
                heading = self.DOWN
                dy, dx = (1, 0)
            elif self.player_heading == self.LEFT:
                heading = self.LEFT
                dy, dx = (0, -1)
            else:
                heading = self.RIGHT
                dy, dx = (0, 1)

            # loop until laser blocking object reached
            ly, lx = (self.player_y, self.player_x)
            while True:
                ly += dy
                lx += dx

                # handle boundary and immovable obstacles
                if ly < 0 or ly >= self.y_size or \
                        lx < 0 or lx >= self.x_size or \
                        self.grid_data[ly][lx] == self.OBSTACLE_SYMBOL:
                    # laser stopped without effect
                    return self.COLLISION

                # handle movable objects
                elif self.cell_is_laser_movable(ly, lx, heading):
                    # check if tile can be moved without collision
                    if self.cell_is_blocked(ly + dy, lx + dx) or \
                            self.grid_data[ly + dy][lx + dx] == self.ICE_SYMBOL or \
                            self.grid_data[ly + dy][lx + dx] == self.TELEPORT_SYMBOL or \
                            self.grid_data[ly + dy][lx + dx] == self.FLAG_SYMBOL or \
                            (ly + dy == self.player_y and lx + dx == self.player_x):
                        # tile cannot be moved
                        return self.COLLISION
                    else:
                        old_symbol = self.grid_data[ly][lx]
                        self.grid_data[ly][lx] = self.LAND_SYMBOL
                        if self.grid_data[ly + dy][lx + dx] == self.WATER_SYMBOL:
                            # if new bridge position is water, convert to land tile
                            if old_symbol == self.BRIDGE_SYMBOL:
                                self.grid_data[ly + dy][lx + dx] = self.LAND_SYMBOL
                            # otherwise, do not replace the old symbol
                        else:
                            # otherwise, move the tile forward
                            self.grid_data[ly + dy][lx + dx] = old_symbol
                        break

                # handle bricks
                elif self.grid_data[ly][lx] == self.BRICK_SYMBOL:
                    # remove brick, replace with land
                    self.grid_data[ly][lx] = self.LAND_SYMBOL
                    break

                # handle facing anti-tanks
                elif (self.grid_data[ly][lx] == self.ANTI_TANK_UP_SYMBOL and heading == self.DOWN) or \
                        (self.grid_data[ly][lx] == self.ANTI_TANK_DOWN_SYMBOL and heading == self.UP) or \
                        (self.grid_data[ly][lx] == self.ANTI_TANK_LEFT_SYMBOL and heading == self.RIGHT) or \
                        (self.grid_data[ly][lx] == self.ANTI_TANK_RIGHT_SYMBOL and heading == self.LEFT):
                    # mark anti-tank as destroyed
                    self.grid_data[ly][lx] = self.ANTI_TANK_DESTROYED_SYMBOL
                    break

                # handle player laser collision
                elif ly == self.player_y and lx == self.player_x:
                    return self.GAME_OVER

                # handle facing mirrors
                elif (self.grid_data[ly][lx] == self.MIRROR_UL_SYMBOL and heading == self.RIGHT) or \
                        (self.grid_data[ly][lx] == self.MIRROR_UR_SYMBOL and heading == self.LEFT):
                    # new direction is up
                    dy, dx = (-1, 0)
                    heading = self.UP
                elif (self.grid_data[ly][lx] == self.MIRROR_DL_SYMBOL and heading == self.RIGHT) or \
                        (self.grid_data[ly][lx] == self.MIRROR_DR_SYMBOL and heading == self.LEFT):
                    # new direction is down
                    dy, dx = (1, 0)
                    heading = self.DOWN
                elif (self.grid_data[ly][lx] == self.MIRROR_UL_SYMBOL and heading == self.DOWN) or \
                        (self.grid_data[ly][lx] == self.MIRROR_DL_SYMBOL and heading == self.UP):
                    # new direction is left
                    dy, dx = (0, -1)
                    heading = self.LEFT
                elif (self.grid_data[ly][lx] == self.MIRROR_UR_SYMBOL and heading == self.DOWN) or \
                        (self.grid_data[ly][lx] == self.MIRROR_DR_SYMBOL and heading == self.UP):
                    # new direction is right
                    dy, dx = (0, 1)
                    heading = self.RIGHT
                # do not terminate laser on facing mirror - keep looping

            # check for game over condition after effect of laser
            if self.cell_is_game_over(self.player_y, self.player_x):
                return self.GAME_OVER

        return self.SUCCESS

    def render(self):
        """
        Render the map's current state to terminal
        """
        for r in range(self.y_size):
            line = ''
            for c in range(self.x_size):
                glyph = self.MAP_GLYPH_TABLE[self.grid_data[r][c]]

                # overwrite with player
                if r == self.player_y and c == self.player_x:
                    glyph = self.PLAYER_GLYPH_TABLE[self.player_heading]

                line += glyph
            print(line)

        print('\n' * (20 - self.y_size))

    def is_finished(self):
        """
        Check if the finish condition (player at flag) has been reached
        :return: True if player at flag, False otherwise
        """
        if self.grid_data[self.player_y][self.player_x] == self.FLAG_SYMBOL:
            return True
        else:
            return False

    def cell_is_blocked(self, y, x):
        """
        Check if the cell with the given coordinates is blocked (i.e. movement
        to this cell is not possible)
        :param y: y coord
        :param x: x coord
        :return: True if blocked, False otherwise
        """
        symbol = self.grid_data[y][x]
        # collision: obstacle, bridge, mirror (all types), anti-tank (all types)
        if symbol == self.OBSTACLE_SYMBOL or symbol == self.BRIDGE_SYMBOL or symbol == self.BRICK_SYMBOL or \
                symbol == self.MIRROR_UL_SYMBOL or symbol == self.MIRROR_UR_SYMBOL or \
                symbol == self.MIRROR_DL_SYMBOL or symbol == self.MIRROR_DR_SYMBOL or \
                symbol == self.ANTI_TANK_UP_SYMBOL or symbol == self.ANTI_TANK_DOWN_SYMBOL or \
                symbol == self.ANTI_TANK_LEFT_SYMBOL or symbol == self.ANTI_TANK_RIGHT_SYMBOL or \
                symbol == self.ANTI_TANK_DESTROYED_SYMBOL:
            return True
        return False

    def cell_is_game_over(self, y, x):
        """
        Check if the cell with the given coordinates will result in game
        over.
        :param y: y coord
        :param x: x coord
        :return: True if blocked, False otherwise
        """
        # check for water
        if self.grid_data[y][x] == self.WATER_SYMBOL:
            return True

        # check for anti-tank
        # up direction
        for i in range(y, -1, -1):
            if self.grid_data[i][x] == self.ANTI_TANK_DOWN_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(i, x):
                break

        # down direction
        for i in range(y, self.y_size):
            if self.grid_data[i][x] == self.ANTI_TANK_UP_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(i, x):
                break

        # left direction
        for i in range(x, -1, -1):
            if self.grid_data[y][i] == self.ANTI_TANK_RIGHT_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(y, i):
                break

        # right direction
        for i in range(x, self.x_size):
            if self.grid_data[y][i] == self.ANTI_TANK_LEFT_SYMBOL:
                return True
            # if blocked, can stop checking for anti-tank
            if self.cell_is_blocked(y, i):
                break

        # no water or anti-tank danger
        return False

    def cell_is_laser_movable(self, y, x, heading):
        """
        Check if the tile at coordinated (y, x) is movable by a laser with the given heading.
        :param y: y coord
        :param x: x coord
        :param heading: laser direction
        :return: True is movable, false otherwise
        """
        return self.grid_data[y][x] in self.MOVABLE_SYMBOLS[heading]


def main(arglist):
    """
    Run a playable game of LaserTank using the given filename as the map file.
    :param arglist: map file name
    """
    try:
        import msvcrt

        def windows_getchar():
            return msvcrt.getch().decode('utf-8')

        getchar = windows_getchar
    except ImportError:
        import tty
        import termios

        def unix_getchar():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        getchar = unix_getchar

    if len(arglist) != 1:
        print("Running this file directly launches a playable game of LaserTank based on the given map file.")
        print("Usage: laser_tank.py [map_file_name]")
        return

    print("Use W to move forward, A and S to turn. Use (spacebar) to shoot. Press 'q' to quit." +
          "Press 'r' to restart the map.")

    map_inst = LaserTankMap.process_input_file(arglist[0])
    map_inst.render()

    steps = 0

    while True:
        char = getchar()

        if char == 'q':
            return

        if char == 'r':
            map_inst = LaserTankMap.process_input_file(arglist[0])
            map_inst.render()

            steps = 0

        if char in ['w', 'a', 'd', ' ']:
            steps += 1
            a = {'w': LaserTankMap.MOVE_FORWARD,
                 'a': LaserTankMap.TURN_LEFT,
                 'd': LaserTankMap.TURN_RIGHT,
                 ' ': LaserTankMap.SHOOT_LASER}[char]

            ret = map_inst.apply_move(a)
            map_inst.render()

            if ret == LaserTankMap.GAME_OVER:
                print('Game Over!')
                return

            if map_inst.is_finished():
                print("Puzzle solved in " + str(steps) + " steps!")
                return


if __name__ == '__main__':
    main(sys.argv[1:])







