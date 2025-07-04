import numpy as np

# ------------------------- #
# Felzenszwalb–Huttenlocher #
# ------------------------- #

# https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf

# Find horizontal intersection between parabolas at the given vertices.
def intersect_parabolas(p, q):
    px, py = p
    qx, qy = q
    x = ((qy + qx*qx) - (py + px*px)) / (2*qx - 2*px)
    return x

def edt_1d(f):
    """
    1D squared‐Euclidean distance transform of array f.
    Returns array output of same shape, where
      output[i] = min_j ( f[j] + (i-j)^2 ).
    """
    n = f.shape[0]
    parabola_xs = np.zeros(n, dtype=np.int32)               # hull parabola locations (x)
    intersects =  np.empty(n+1, dtype=np.float64)           # hull parabola intersects
    output =      np.empty(n, dtype=f.dtype)                # output row/col

    parabola_xs[0] = 0
    intersects[0] = -1e20
    intersects[1] = +1e20

    # Build lower envelope
    k = 0
    for x in range(1, n):
        par_a = (x, f[x])
        par_b = (parabola_xs[k], f[parabola_xs[k]])
        int_x = intersect_parabolas(par_a, par_b)
        while int_x <= intersects[k]:
            k -= 1
            par_b = (parabola_xs[k], f[parabola_xs[k]])
            int_x = intersect_parabolas(par_a, par_b)
        k += 1
        parabola_xs[k]  = x
        intersects[k]   = int_x
        intersects[k+1] = 1e20

    # Compute distances
    k = 0
    for x in range(n):
        while intersects[k+1] < x:
            k += 1
        dx = x - parabola_xs[k]
        output[x] = dx*dx + f[parabola_xs[k]]

    return output

def distance_transform_2d_lower(f):
    """
    2D squared‐Euclidean distance transform of 2D array f.
    Returns array d of same shape:
      d[i,j] = min_{p,q} ( f[p,q] + (i-p)^2 + (j-q)^2 ).
    """
    H, W = f.shape
    # First pass: transform each row
    temp = np.empty_like(f, dtype=np.float64)
    for y in range(H):
        temp[y, :] = edt_1d(f[y, :])
    
    test = np.sqrt(temp)
    print("pass 1 ^2: \n", temp)
    print("pass 1: \n", test)

    # Second pass: transform each column of the intermediate
    d2 = np.empty_like(temp, dtype=np.float64)
    for x in range(W):
        d2[:, x] = edt_1d(temp[:, x])

    print("pass 2 ^2: \n", d2)

    return d2

def distance_transform_2d_upper(f):
    g = -f
    d = -distance_transform_2d_lower(g)
    return d

# Example usage:
if __name__ == "__main__":
    # Example: binary image with a single zero at (4,0) and infinities elsewhere
    f = np.full((9,9), 4.0, dtype=np.float32)
    # input_img[2,1] = 0.0
    f[0,0] = 0.0
    print("input: \n", f)

    f = f*f

    # Compute squared distances
    d2 = distance_transform_2d_lower(f)

    # If you need actual Euclidean distances:
    d = np.sqrt(d2)

    # Print rounded result
    print("output: \n", np.round(d, 3))
    # Expected:
    # [[8.    8.123 8.472 9.    9.657]
    #  [7.    7.162 7.606 8.243 9.   ]
    #  [6.    6.236 6.828 7.606 8.472]
    #  [5.    5.414 6.236 7.162 8.123]
    #  [4.    5.    6.    7.    8.   ]]
