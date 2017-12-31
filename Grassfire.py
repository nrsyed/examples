import numpy as np
import random
import math

PI = math.pi

class Grassfire:
    '''Class is a container for constants and methods we'll use
        to create, modify, and plot a 2D grid of pixels
        demonstrating the grassfire path-planning algorithm.
    '''

    START = 0
    DEST = -1     # destination
    UNVIS = -2    # unvisited
    OBST = -3     # obstacle
    PATH = -4

    # Each of the above cell values is represented by an RGB color
    # on the plot. COLOR_VIS refers to visited cells (value > 0).
    COLOR_START = np.array([0, 0.75, 0])
    COLOR_DEST = np.array([0.75, 0, 0])
    COLOR_UNVIS = np.array([1, 1, 1])
    COLOR_VIS = np.array([0, 0.5, 1])
    COLOR_OBST = np.array([0, 0, 0])
    COLOR_PATH = np.array([1, 1, 0])

    def random_grid(cls, rows=16, cols=16, obstacleProb=0.3):
        '''Return a 2D numpy array representing a grid of randomly placed
        obstacles (where the likelihood of any cell being an obstacle
        is given by obstacleProb) and randomized start/destination cells.
        '''
        obstacleGrid = np.random.random_sample((rows, cols))
        grid = cls.UNVIS * np.ones((rows, cols), dtype=np.int)
        grid[obstacleGrid <= obstacleProb] = cls.OBST

        # Randomly set start and destination cells.
        cls.set_start_dest(grid)
        return grid

    def set_start_dest(cls, grid):
        '''For a given grid, randomly select start and destination cells.'''
        (rows, cols) = grid.shape

        # Remove existing start and dest cells, if any.
        grid[grid == cls.START] = cls.UNVIS
        grid[grid == cls.DEST] = cls.UNVIS

        # Randomize start cell.
        validStartCell = False
        while not validStartCell:
            startIndex = random.randint(0, rows * cols - 1)
            startIndices = np.unravel_index(startIndex, (rows, cols))
            if grid[startIndices] != cls.OBST:
                validStartCell = True
                grid[startIndices] = cls.START

        # Randomize destination cell.
        validDestCell = False
        while not validDestCell:
            destIndex = random.randint(0, rows * cols - 1)
            destIndices = np.unravel_index(destIndex, (rows, cols))
            if grid[destIndices] != cls.START and grid[destIndices] != cls.OBST:
                validDestCell = True
                grid[destIndices] = cls.DEST

    def color_grid(cls, grid):
        '''Return MxNx3 pixel array ("color grid") corresponding to a grid.'''
        (rows, cols) = grid.shape
        colorGrid = np.zeros((rows, cols, 3), dtype=np.float)

        colorGrid[grid == cls.OBST, :] = cls.COLOR_OBST
        colorGrid[grid == cls.UNVIS, :] = cls.COLOR_UNVIS
        colorGrid[grid == cls.START, :] = cls.COLOR_START
        colorGrid[grid == cls.DEST, :] = cls.COLOR_DEST
        colorGrid[grid > cls.START, :] = cls.COLOR_VIS
        colorGrid[grid == cls.PATH, :] = cls.COLOR_PATH
        return colorGrid

    def reset_grid(cls, grid):
        '''Reset cells that are not OBST, START, or DEST to UNVIS.'''
        cellsToReset = ~((grid == cls.OBST) + (grid == cls.START)
            + (grid == cls.DEST))
        grid[cellsToReset] = cls.UNVIS

    def _check_adjacent(cls, grid, cell, currentDepth):
        '''For given grid, check the cells adjacent to a given
            cell. If any have a depth (positive int) greater
            than the current depth, update them with the current
            depth, where depth represents distance from start cell.
            If destination found, return DEST constant; else, return
            number of adjacent cells updated.
        '''
        (rows, cols) = grid.shape

        # Track how many adjacent cells are updated.
        numCellsUpdated = 0

        # From the current cell, examine, using sin and cos:
        # cell to right (col + 1), cell below (row + 1),
        # cell to left (col - 1), cell above (row - 1).
        for i in range(4):
            rowToCheck = cell[0] + int(math.sin((PI/2) * i))
            colToCheck = cell[1] + int(math.cos((PI/2) * i))

            # Ensure cell is within bounds of grid.
            if not (0 <= rowToCheck < rows and 0 <= colToCheck < cols):
                continue
            # Check if destination found.
            elif grid[rowToCheck, colToCheck] == cls.DEST:
                return cls.DEST
            # If adjacent cell unvisited or depth > currentDepth + 1,
            # mark with new depth.
            elif (grid[rowToCheck, colToCheck] == cls.UNVIS
                or grid[rowToCheck, colToCheck] > currentDepth + 1):
                grid[rowToCheck, colToCheck] = currentDepth + 1
                numCellsUpdated += 1
        return numCellsUpdated

    def _backtrack(cls, grid, cell, currentDepth):
        '''This function is used if the destination is found. Similar
            to _check_adjacent(), but returns coordinates of first
            surrounding cell whose value matches "currentDepth", ie,
            the next cell along the path from destination to start.
        '''
        (rows, cols) = grid.shape

        for i in range(4):
            rowToCheck = cell[0] + int(math.sin((PI/2) * i))
            colToCheck = cell[1] + int(math.cos((PI/2) * i))

            if not (0 <= rowToCheck < rows and 0 <= colToCheck < cols):
                continue
            elif grid[rowToCheck, colToCheck] == currentDepth:
                nextCell = (rowToCheck, colToCheck)
                grid[nextCell] = cls.PATH
                return nextCell

    def find_path(cls, grid):
        '''Execute grassfire algorithm by spreading from the start cell out.
            If destination is found, use _backtrack() to trace path from
            destination back to start. Returns a generator function to
            allow stepping through and animating the algorithm.
        '''
        nonlocalDict = {'grid': grid}
        def find_path_generator():
            grid = nonlocalDict['grid']
            depth = 0
            destFound = False
            cellsExhausted = False

            while (not destFound) and (not cellsExhausted):
                numCellsModified = 0
                depthIndices = np.where(grid == depth)
                matchingCells = list(zip(depthIndices[0], depthIndices[1]))

                for cell in matchingCells:
                    adjacentVal = cls._check_adjacent(grid, cell, depth)
                    if adjacentVal == cls.DEST:
                        destFound = True
                        break
                    else:
                        numCellsModified += adjacentVal

                if numCellsModified == 0:
                    cellsExhausted = True
                elif not destFound:
                    depth += 1
                yield

            if destFound:
                destCell = np.where(grid == cls.DEST)
                backtrackCell = (destCell[0].item(), destCell[1].item())
                while depth > 0:
                    # Work backwards until return to start cell.
                    nextCell = cls._backtrack(grid, backtrackCell, depth)
                    backtrackCell = nextCell
                    depth -= 1
                    yield
        return find_path_generator
