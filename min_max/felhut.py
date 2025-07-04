import math
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

# ------------------------- #
# Felzenszwalb–Huttenlocher #
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

def run_square_shader(program, texture_handle, mode, width, height):
    glUseProgram(program)
    
    # Bind input/output texture to binding 0
    glBindImageTexture(0, texture_handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F)

    # Set mode uniform: 0 = square, 1 = sqrt
    glUniform1i(glGetUniformLocation(program, "uMode"), mode)

    # Dispatch with same local size as in shader (16x16)
    groups_x = (width + 15) // 16
    groups_y = (height + 15) // 16
    glDispatchCompute(groups_x, groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)



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

    input_img = np.full((5,9), 4.0, dtype=np.float32)
    input_img[0,0] = 0.0
    H, W = input_img.shape

    # input_img = input_img * input_img
    
    texA          = save_texture(input_img)                                 # original A
    texNext       = save_texture(input_img.copy())                          # output buffer


    # ------------------------------------------------------------
    # 3. Compile compute shader via helpers
    # ------------------------------------------------------------
    with open("felhut.comp", "r", encoding="utf-8") as f:
        felhut_src = f.read()

    with open("square.comp", "r", encoding="utf-8") as f:
        square_src = f.read()

    prog_felhut = compileProgram(
        compileShader(felhut_src, GL_COMPUTE_SHADER)
    )

    prog_square = compileProgram(
        compileShader(square_src, GL_COMPUTE_SHADER)
    )

    # ---- INIT UNIFORMS ----
    loc_scan = glGetUniformLocation(prog_felhut, "uScanID")
    loc_mode = glGetUniformLocation(prog_felhut, "uMode")
    loc_size   = glGetUniformLocation(prog_felhut, "uSize")
    loc_stride = glGetUniformLocation(prog_felhut, "uStride")

    # ---- INIT SSBOS ----
    parabola_xs     = np.zeros((H + 1, W + 1), dtype=np.int32)
    intersects      = np.zeros((H + 1, W + 1), dtype=np.float32)

    parabola_ssbo   = save_ssbo(parabola_xs, binding_index=1)
    intersects_ssbo = save_ssbo(intersects, binding_index=2)

    # ------------------------------------------------------------
    # 4. itterate passes: scan rows, scan cols
    # ------------------------------------------------------------

    def setup_felhut_shader(scanID):
        glUseProgram(prog_felhut)
        glUniform1i(loc_scan, scanID)
        glUniform1i(loc_mode, 1)    # 0=lower, 1=upper
        glUniform2i(loc_size, W, H)
        # textures
        glBindImageTexture(0, texNext, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F)
    
    for scanID in range(4):


        if scanID == 0:
            # SQUARE
            run_square_shader(prog_square, texNext, 0, W, H)

        elif scanID == 1:
            # ROWS
            setup_felhut_shader(0)
            STRIDE = W + 1
            glUniform1i(loc_stride, STRIDE)
            # row‑scans: gl_WorkGroupID.y ∈ [0..H)
            glDispatchCompute(1, H, 1)
        elif scanID == 2:
            # COLS
            setup_felhut_shader(1)
            STRIDE = H + 1
            glUniform1i(loc_stride, STRIDE)
            # col‑scans: gl_WorkGroupID.x ∈ [0..W)
            glDispatchCompute(W, 1, 1)
        else:
            # SQRT
            run_square_shader(prog_square, texNext, 1, W, H)


        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


        # # read result
        # glBindTexture(GL_TEXTURE_2D, texNext)
        # raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
        # result = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        # print(np.round(result, 3))

    

    # ------------------------------------------------------------
    # 5. Read result
    # ------------------------------------------------------------
    glBindTexture(GL_TEXTURE_2D, texNext)
    raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    result = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    # result = np.sqrt(result)
    print(np.round(result, 3))

    # ------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------
    glfw.terminate()

