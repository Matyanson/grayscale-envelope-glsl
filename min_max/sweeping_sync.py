import math
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

# ------------------------- #
#  FAST SWEEPING SEQUENTIAL #
# ------------------------- #


def getDist(x, y):
    # return abs(x) + abs(y) # Manhattan
    # return math.sqrt(x*x + y*y) # Euclidean 
    return max(abs(x), abs(y)) # Chebyshev


# Example usage
if __name__ == "__main__":

    # Load input image
    img = Image.open("input/grey.png").convert("L")
    A = np.array(img, dtype=np.float64)

    input_img = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [4,0,0,0,0],
    ], dtype=np.float32)
    H, W = input_img.shape

    # Sweep directions (4 quadrants)
    sweep_directions = [
        (range(H), range(W)),                         # TL -> BR
        (range(H-1, -1, -1), range(W-1, -1, -1)),     # BR -> TL
        (range(H), range(W-1, -1, -1)),               # TR -> BL
        (range(H-1, -1, -1), range(W)),               # BL -> TR
    ]

    #   Define which neighbor offsets to consider this pass:
    #   sweep 0 = TL->BR  reads North(0,-1), West(-1,0), NW(-1,-1), NE(+1,-1)
    #   sweep 1 = BR->TL  reads South(0,+1), East(+1,0), SE(+1,+1), SW(-1,+1)
    #   sweep 2 = TR->BL  reads North(0,-1), East(+1,0), NE(+1,-1), SE(+1,+1)
    #   sweep 3 = BL->TR  reads South(0,+1), West(-1,0), SW(-1,+1), NW(-1,-1)
    neighbours = [
        [( 0,-1), (-1, 0), (-1,-1), ( 1,-1)],
        [( 0, 1), ( 1, 0), ( 1, 1), (-1, 1)],
        [( 0,-1), ( 1, 0), ( 1,-1), ( 1, 1)],
        [( 0, 1), (-1, 0), (-1, 1), (-1,-1)]
    ]

    def update(i, j, sweep):
        old = input_img[j][i]
        best = old
        for k in range(3):
            dx, dy = neighbours[sweep][k]
            i2 = i + dx
            j2 = j + dy
            if 0 <= i2 < W and 0 <= j2 < H:
                val = input_img[j2][i2]
                cost = getDist(dx, dy)
                best = max(best, val - cost)
        input_img[j][i] = best
        return best != old

    

    for sweep in range(4):
        print("sweep: ", sweep)
        dir_h, dir_w = sweep_directions[sweep]
        for j in dir_h:
            for i in dir_w:
                update(i, j, sweep)
        print(input_img)

    # converged = False
    # iteration = 0
    # while not converged:
    #     converged = True
    #     for sweep in range(4):
    #         print("sweep: ", sweep)
    #         changed = False
    #         dir_h, dir_w = sweep_directions[sweep]
    #         for j in dir_h:
    #             for i in dir_w:
    #                 if update(i, j, sweep):
    #                     changed = True
    #         if changed:
    #             converged = False
    #         print(input_img)

    #     iteration += 1
    #     print(f"Iteration {iteration}:\n", input_img)
