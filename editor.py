import sys
import math
import random
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
from engine.engine import Camera, Transform, TriangleObject, load_obj, create_projection_matrix

# Vertex shader
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec4 position;
uniform mat4 mvp;
void main() {
    gl_Position = mvp * position;
}
"""

# Fragment shader
fragment_shader_source = """
#version 330 core
uniform vec3 color;
out vec4 out_color;
void main() {
    out_color = vec4(color, 1.0);
}
"""

def main():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create window
    window = glfw.create_window(800, 600, "PyPointEngine", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    # Initialize OpenGL
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Compile shaders
    try:
        shader = compileProgram(
            compileShader(vertex_shader_source, GL_VERTEX_SHADER),
            compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        )
    except RuntimeError as e:
        print(f"Shader compilation failed: {e}")
        glfw.terminate()
        return

    # Load models (load each OBJ file only once)
    tobjects = []
    try:
        # Load svistok.obj once
        vertices_svistok, faces_svistok = load_obj("resources/svistok.obj", scale=0.5)
        print(f"Svistok: {len(vertices_svistok)} vertices, {len(faces_svistok)} faces")
        indices_svistok = [i for face in faces_svistok for i in face]
        
        # Create 100 instances with different positions
        x, y = 0, 10
        baka = False
        for _ in range(100):
            if baka == True:
                x += 3
                y += 0.5
                obj = TriangleObject(vertices_svistok, indices_svistok, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                obj.transform.position = (x, y, 0)  # Apply offset via Transform
                tobjects.append(obj)
                baka = False
            else:
                x += 3
                y = y * -0.1
                obj = TriangleObject(vertices_svistok, indices_svistok, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                obj.transform.position = (x, y, 0)  # Apply offset via Transform
                tobjects.append(obj)
                baka = True

        # Load sphere.obj
        vertices, faces = load_obj("resources/sphere.obj", scale=0.5, offset=(-6, 0, 0))
        print(f"Sphere: {len(vertices)} vertices, {len(faces)} faces")
        indices = [i for face in faces for i in face]
        tobjects.append(TriangleObject(vertices, indices, color=(0, 1, 1)))

        # Load model.obj
        vertices, faces = load_obj("resources/model.obj", scale=0.5, offset=(-9, 0, 0))
        print(f"Model: {len(vertices)} vertices, {len(faces)} faces")
        indices = [i for face in faces for i in face]
        tobjects.append(TriangleObject(vertices, indices, color=(0, 1, 1)))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        glfw.terminate()
        return

    # Setup OpenGL buffers
    import OpenGL.GL as gl
    for obj in tobjects:
        obj.setup_buffers(gl)

    # Setup camera
    width, height = glfw.get_framebuffer_size(window)
    cam = Camera(fov=70, aspect=width/height, near=0.1, far=100.0)
    cam.transform.position = (0.0, 0.0, -10.0)
    proj = create_projection_matrix(cam.fov, cam.aspect, cam.near, cam.far)
    view = cam.get_view_matrix()

    # Mouse control
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    mouse_sens = 0.003
    last_x, last_y = width / 2, height / 2
    first_mouse = True

    # FPS counter
    fps = 0
    frame_count = 0
    last_time = glfw.get_time()
    fps_update_interval = 1.0

    def framebuffer_size_callback(window, width, height):
        nonlocal proj, cam
        glViewport(0, 0, width, height)
        cam.aspect = width / height
        proj = create_projection_matrix(cam.fov, cam.aspect, cam.near, cam.far)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        dt = current_time - last_time if frame_count > 0 else 1/60
        last_time = current_time

        # Handle input
        move = [0, 0, 0]
        speed = 5.0 * dt
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            move[2] -= speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            move[2] += speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            move[0] -= speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            move[0] += speed
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            move[1] -= speed
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            move[1] += speed
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 100
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # Mouse look
        x, y = glfw.get_cursor_pos(window)
        if first_mouse:
            last_x, last_y = x, y
            first_mouse = False
        mx, my = x - last_x, last_y - y
        last_x, last_y = x, y

        pitch, yaw, roll = cam.transform.rotation
        new_rotation = (
            max(-math.pi/2, min(math.pi/2, pitch + my * mouse_sens)),
            yaw - mx * mouse_sens,
            roll
        )

        # Compute camera movement
        pitch, yaw, _ = new_rotation
        forward = (math.sin(yaw), 0, math.cos(yaw))
        right = (math.cos(yaw), 0, -math.sin(yaw))
        up = (0, 1, 0)

        new_position = (
            cam.transform.position[0] + forward[0]*move[2] + right[0]*move[0],
            cam.transform.position[1] + move[1],
            cam.transform.position[2] + forward[2]*move[2] + right[2]*move[0]
        )

        # Update view matrix
        if new_position != cam.transform.position or new_rotation != cam.transform.rotation:
            cam.transform.position = new_position
            cam.transform.rotation = new_rotation
            view = cam.get_view_matrix()

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render objects
        glUseProgram(shader)
        mvp_loc = glGetUniformLocation(shader, "mvp")
        color_loc = glGetUniformLocation(shader, "color")

        for obj in tobjects:
            model = obj.transform.get_matrix()
            mvp = np.dot(np.dot(proj, view), model)
            glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, np.array(mvp, dtype=np.float32))
            glUniform3fv(color_loc, 1, obj.color)
            glBindVertexArray(obj.vao)
            glDrawElements(GL_TRIANGLES, len(obj.indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

        # FPS counter
        frame_count += 1
        if current_time - last_time >= fps_update_interval:
            fps = frame_count / (current_time - last_time)
            frame_count = 0
            last_time = current_time
            print(f"FPS: {fps:.1f}")

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()