import math
import os
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

# ------------------------- #
#       MORPHOLOGICAL       #
# ------------------------- #

def save_texture(data=None):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, W, H)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    if data is not None:
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H,
                        GL_RED, GL_FLOAT, data)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex

def save_ssbo(array, binding_index):
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, array.nbytes, array, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_index, ssbo)
    return ssbo

def copy_ssbo(buffer_id, dtype, count):
    if dtype == np.int32:       ctype_arr = ctypes.c_int32 * count
    elif dtype == np.uint32:       ctype_arr = ctypes.c_uint32 * count
    elif dtype == np.float32:   ctype_arr = ctypes.c_float * count
    else:                       raise ValueError(f"Unsupported dtype: {dtype}")
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY)
    buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctype_arr))
    result = np.frombuffer(buffer_ptr.contents, dtype=dtype).copy()
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    return result

def run_morph(A, mode, max_gradient = 1.0, dist_method = 0):
    input_img = A
    texA    = save_texture(input_img)          # original A
    texPrev = save_texture(input_img.copy())   # envelope ping-pong buffer
    texNext = save_texture()                   # temp buffer

    # ------------------------------------------------------------
    # 3. Compile compute shader via helpers
    # ------------------------------------------------------------
    with open("morph.comp", "r", encoding="utf-8") as f:
        compute_src = f.read()

    prog = compileProgram(
        compileShader(compute_src, GL_COMPUTE_SHADER)
    )

    # ---- INIT UNIFORMS ----
    loc_mode = glGetUniformLocation(prog, "uMode")
    loc_dist_method = glGetUniformLocation(prog, "uDistMethod")
    loc_max_gradient = glGetUniformLocation(prog, "uMaxGradient")

    # ------------------------------------------------------------
    # 4. itterate passes: dispatch compute, ping-pong buffers
    # ------------------------------------------------------------
    if dist_method == 1:  sweeps_needed = math.ceil(max(W, H) / max_gradient) 
    else:               sweeps_needed = math.ceil((W + H) / max_gradient)

    # uniforms
    glUseProgram(prog)
    glUniform1i(loc_mode, mode)
    glUniform1i(loc_dist_method, dist_method)
    glUniform1f(loc_max_gradient, max_gradient)
    glBindImageTexture(0, texA,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
    # SSBOS
    flag_ssbo = save_ssbo(np.array([0], dtype=np.uint32), 3)
    
    for i in range(sweeps_needed):
        # bind images
        glBindImageTexture(1, texPrev, 0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
        glBindImageTexture(2, texNext, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F)

        # dispatch
        gx = (W + 15) // 16
        gy = (H + 15) // 16
        glDispatchCompute(gx, gy, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)

        # swap ping-pong
        texPrev, texNext = texNext, texPrev

        # check for changes
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, flag_ssbo)
        flag_value = copy_ssbo(flag_ssbo, np.uint32, 1)

        if flag_value[0] == 0:
            print(f"converged after {i} passes")
            break

        # clear the flag
        zero = np.array(0, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, flag_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, zero)

    

    # ------------------------------------------------------------
    # 5. Read result
    # ------------------------------------------------------------
    glBindTexture(GL_TEXTURE_2D, texNext)
    raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)

    return arr

def run_average(A, B):
    texA    = save_texture(A)          # original A
    texB    = save_texture(B)          # original B

    # ------------------------------------------------------------
    # 3. Compile compute shader via helpers
    # ------------------------------------------------------------
    with open("average.comp", "r", encoding="utf-8") as f:
        compute_src = f.read()

    prog = compileProgram(
        compileShader(compute_src, GL_COMPUTE_SHADER)
    )

    # ---- INIT UNIFORMS ----
    loc_mode = glGetUniformLocation(prog, "uMode")

    # ------------------------------------------------------------
    # 4. itterate passes
    # ------------------------------------------------------------

    # uniforms
    glUseProgram(prog)
    glUniform1i(loc_mode, 0)
    glBindImageTexture(0, texA,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
    glBindImageTexture(1, texB,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)

    # dispatch
    gx = (W + 15) // 16
    gy = (H + 15) // 16
    glDispatchCompute(gx, gy, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # ------------------------------------------------------------
    # 5. Read result
    # ------------------------------------------------------------
    glBindTexture(GL_TEXTURE_2D, texA)
    raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)

    return arr



# Example usage
if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. Boilerplate: init GLFW + create invisible window + GL ctx
    # ------------------------------------------------------------
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    # Request an OpenGL 4.3+ context for compute shaders
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(16, 16, "offscreen", None, None)
    glfw.make_context_current(window)

    # ------------------------------------------------------------
    # 2. Prepare input data (5×5 example), upload to texture A
    # ------------------------------------------------------------

    # Load input image
    img = Image.open("input/grey.png").convert("L")
    A = np.array(img, dtype=np.float64)
    H, W = A.shape

    A_low = np.maximum(A, 0)
    A_up = np.minimum(A, 255)

    max_gradient = 20.0 * 0.70710678118
    dist_method = 2
    lower_arr = run_morph(A_low, 0, max_gradient, dist_method)
    upper_arr = run_morph(A_up, 1, max_gradient, dist_method)

    lower_img = Image.fromarray(np.clip(lower_arr, 0, 255).astype(np.uint8), mode="L")
    lower_img.save(os.path.join("output", "lower_envelope.png"))

    upper_img = Image.fromarray(np.clip(upper_arr, 0, 255).astype(np.uint8), mode="L")
    upper_img.save(os.path.join("output", "upper_envelope.png"))

    average_arr = run_average(lower_arr, upper_arr)
    average_img = Image.fromarray(np.clip(average_arr, 0, 255).astype(np.uint8), mode="L")
    average_img.save(os.path.join("output", "average_envelope.png"))


    # ------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------
    glfw.terminate()
