import heapq
import math
import os
import numpy as np
from PIL import Image

# ------------------------- #
#   MULTISOURCE DIJKSTRA    #
# ------------------------- #

def euclid_envelope(A, lower=True):
    """
    A: 2D numpy array (float or int)
    lower: if True, compute L = min(A(u,v)+dist);
           if False, compute U = max(A(u,v)-dist).
    Returns: 2D numpy array of envelope.
    """
    H, W = A.shape
    # If computing upper, negate A
    if not lower:
        A = -A

    # dist array: initialize to +inf
    dist = np.full((H, W), np.inf, dtype=float)
    # visited mask
    seen = np.zeros((H, W), bool)

    # Min-heap of (cost, i, j)
    heap = []
    for i in range(H):
        for j in range(W):
            dist[i,j] = A[i,j]
            heapq.heappush(heap, (dist[i,j], i, j))

    # 8-connected moves
    moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
             (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
             (1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]

    while heap:
        d, i, j = heapq.heappop(heap)
        if seen[i,j]:
            continue
        seen[i,j] = True
        # Relax neighbors
        for di, dj, w in moves:
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W:
                nd = d + w
                if nd < dist[ni,nj]:
                    dist[ni,nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))

    # If upper envelope, negate back
    if not lower:
        dist = -dist
    return dist

def map_to_range(arr, source_range, target_range):
    arr = (arr * target_range) / source_range
    arr = arr.astype(np.int32)
    return arr

# Example usage:
img = Image.open("input/grey.png").convert("L")
A = np.array(img, float)
# A = map_to_range(A, 255, 10)

L = euclid_envelope(A, lower=True)
# L = map_to_range(L, 10, 255)
U = euclid_envelope(A, lower=False)
# U = map_to_range(U, 10, 255)

# Clip to [0,255] and convert
L_img = Image.fromarray(np.clip(L,0,255).astype(np.uint8), "L")
U_img = Image.fromarray(np.clip(U,0,255).astype(np.uint8), "L")
# Average
AVG_img = Image.fromarray(((L + U)/2).clip(0,255).round().astype(np.uint8), "L")

os.makedirs("output", exist_ok=True)
L_img.save("output/lower_envelope.png")
U_img.save("output/upper_envelope.png")
AVG_img.save("output/average_envelope.png")
