import math
import os
import numpy as np
from PIL import Image

# If I want the pyramid sides to be at 45deg, the diagonal needs to be 2sqrt(2). (usually it is 2)
# Now that d=2sqrt(2) scale pyramid back to d=2 (* 1 / sqrt(2)). The height is 1 / sqrt(2) = sqrt(2) / 2.

# max_height_dif = 1.0
max_height_dif = math.sqrt(2) / 2

def lower_envelope_1d(f: np.ndarray) -> np.ndarray:
    n = len(f)
    g = f.astype(np.float32).copy()
    for i in range(1, n):
        g[i] = min(g[i], g[i-1] + max_height_dif)
    for i in range(n-2, -1, -1):
        g[i] = min(g[i], g[i+1] + max_height_dif)
    return g

def upper_envelope_1d(f: np.ndarray) -> np.ndarray:
    n = len(f)
    g = f.astype(np.float32).copy()
    for i in range(1, n):
        g[i] = max(g[i], g[i-1] - max_height_dif)
    for i in range(n-2, -1, -1):
        g[i] = max(g[i], g[i+1] - max_height_dif)
    return g

def lower_envelope_2d(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.int32)
    for i in range(arr.shape[0]):
        arr[i, :] = lower_envelope_1d(arr[i, :])
    for j in range(arr.shape[1]):
        arr[:, j] = lower_envelope_1d(arr[:, j])
    return arr

def upper_envelope_2d(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.int32)
    for i in range(arr.shape[0]):
        arr[i, :] = upper_envelope_1d(arr[i, :])
    for j in range(arr.shape[1]):
        arr[:, j] = upper_envelope_1d(arr[:, j])
    return arr

def map_to_range(arr, source_range, target_range):
    arr = (arr * target_range) / source_range
    arr = arr.astype(np.int32)
    return arr

if __name__ == "__main__":
    input_path = "input/grey.png"
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = map_to_range(arr, 255, 10)

    lower_arr = lower_envelope_2d(arr)
    lower_arr = map_to_range(lower_arr, 10, 255)
    upper_arr = upper_envelope_2d(arr)
    upper_arr = map_to_range(upper_arr, 10, 255)

    lower_img = Image.fromarray(np.clip(lower_arr, 0, 255).astype(np.uint8), mode="L")
    upper_img = Image.fromarray(np.clip(upper_arr, 0, 255).astype(np.uint8), mode="L")

    average_arr = ((lower_arr + upper_arr) / 2).round().astype(np.uint8)
    average_img = Image.fromarray(average_arr, mode="L")

    lower_img.save(os.path.join(out_dir, "lower_envelope.png"))
    upper_img.save(os.path.join(out_dir, "upper_envelope.png"))
    average_img.save(os.path.join(out_dir, "average_envelope.png"))

    print("Saved:")
    print(f"  {out_dir}/lower_envelope.png")
    print(f"  {out_dir}/upper_envelope.png")
    print(f"  {out_dir}/average_envelope.png")
