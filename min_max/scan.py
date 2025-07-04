import math
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import glfw

# ------------------------- #
#     2D ENVELOPE SCAN      #
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

    input_img = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [4,0,0,0,0],
    ], dtype=np.float32)
    H, W = input_img.shape

    texA    = save_texture(input_img)          # original A
    texNext = save_texture(input_img.copy())   # output buffer

    # ------------------------------------------------------------
    # 3. Compile compute shader via helpers
    # ------------------------------------------------------------
    with open("envelope_scan.comp", "r", encoding="utf-8") as f:
        compute_src = f.read()

    prog = compileProgram(
        compileShader(compute_src, GL_COMPUTE_SHADER)
    )

    # ---- INIT UNIFORMS ----
    loc_scan = glGetUniformLocation(prog, "uScanID")
    loc_mode = glGetUniformLocation(prog, "uMode")
    loc_dist_method = glGetUniformLocation(prog, "uDistMethod")
    loc_size   = glGetUniformLocation(prog, "uSize")

    # ------------------------------------------------------------
    # 4. itterate passes: dispatch compute, ping-pong buffers
    # ------------------------------------------------------------
    is_chebyshev = False

    glUseProgram(prog)
    glUniform1i(loc_mode, 1)    # 0=lower, 1=upper
    glUniform1i(loc_dist_method, 1 if is_chebyshev else 0)
    glUniform2i(loc_size, W, H)
    glBindImageTexture(0, texA,    0, GL_FALSE, 0, GL_READ_ONLY,  GL_R32F)
    glBindImageTexture(2, texNext, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F)
    
    for scanID in range(2):

        glUniform1i(loc_scan, scanID)

        # dispatch: use one workgroup per row or per column
        if scanID < 1:
            # row‑scans: gl_WorkGroupID.y ∈ [0..H)
            glDispatchCompute(1, H, 1)
        else:
            # col‑scans: gl_WorkGroupID.x ∈ [0..W)
            glDispatchCompute(W, 1, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # swap ping-pong
        # texPrev, texNext = texNext, texPrev

        # read result
        glBindTexture(GL_TEXTURE_2D, texNext)
        raw = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
        result = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        print(np.round(result, 5))

    

    # ------------------------------------------------------------
    # 5. Read result
    # ------------------------------------------------------------
    

    # ------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------
    glfw.terminate()
