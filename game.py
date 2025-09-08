import pygame
import sys
import math
import numpy as np

from OpenGL.GL import *
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader

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


# Function to render Pygame surface as OpenGL texture
def render_text_to_texture(text_surface):
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    width, height = text_surface.get_size()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    return texture_id, width, height


def draw_textured_quad(texture_id, width, height, x, y):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y)
    glTexCoord2f(1, 0); glVertex2f(x + width, y)
    glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
    glTexCoord2f(0, 1); glVertex2f(x, y + height)
    glEnd()

    glDisable(GL_TEXTURE_2D)


# Initialize Pygame and OpenGL
game_width, game_height = 800, 600
pygame.init()
screen = pygame.display.set_mode((game_width, game_height), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("KLGAMEENGINE")

glClearColor(0.1, 0.1, 0.1, 1.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 20)


# Compile shaders
try:
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )
except RuntimeError as e:
    print(f"Shader compilation failed: {e}")
    pygame.quit()
    sys.exit(1)


# Load models
tobjects = []
try:
    vertices3, faces3 = load_obj("resources/svistok.obj", scale=0.5, offset=(-2, 10, 0))
    print(f"Sphere 1: {len(vertices3)} vertices, {len(faces3)} faces")
    indices3 = []
    for face in faces3:
        indices3.extend(face)
    tobjects.append(TriangleObject(vertices3, indices3, color=(0, 255, 255)))

    vertices3, faces3 = load_obj("resources/sphere.obj", scale=0.5, offset=(-6, 0, 0))
    print(f"Sphere 2: {len(vertices3)} vertices, {len(faces3)} faces")
    indices3 = []
    for face in faces3:
        indices3.extend(face)
    tobjects.append(TriangleObject(vertices3, indices3, color=(0, 255, 255)))

    vertices3, faces3 = load_obj("resources/model.obj", scale=0.5, offset=(-9, 0, 0))
    print(f"Sphere 3: {len(vertices3)} vertices, {len(faces3)} faces")
    indices3 = []
    for face in faces3:
        indices3.extend(face)
    tobjects.append(TriangleObject(vertices3, indices3, color=(0, 255, 255)))

except FileNotFoundError as e:
    print(f"Error: {e}")
    pygame.quit()
    sys.exit(1)


# Setup OpenGL buffers
for obj in tobjects:
    obj.setup_buffers(gl)


# Setup camera
cam = Camera(fov=70, aspect=game_width/game_height, near=0.1, far=100.0)
cam.transform.position = (0.0, 0.0, -10.0)

proj = create_projection_matrix(cam.fov, cam.aspect, cam.near, cam.far)
view = cam.get_view_matrix()


# Mouse control
pygame.event.set_grab(True)
pygame.mouse.set_visible(False)
mouse_sens = 0.003


# FPS counter
fps = 0
last_time = pygame.time.get_ticks()
frame_count = 0
fps_update_interval = 1000


while True:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        pygame.quit()
        sys.exit()

    # Mouse look
    mx, my = pygame.mouse.get_rel()
    pitch, yaw, roll = cam.transform.rotation
    new_rotation = (
        max(-math.pi/2, min(math.pi/2, pitch - my * mouse_sens)),
        yaw - mx * mouse_sens,
        roll
    )


    # Movement
    speed = 5.0 * dt
    move = [0, 0, 0]
    if keys[pygame.K_s]: move[2] += speed
    if keys[pygame.K_w]: move[2] -= speed
    if keys[pygame.K_a]: move[0] -= speed
    if keys[pygame.K_d]: move[0] += speed
    if keys[pygame.K_q]: move[1] -= speed
    if keys[pygame.K_e]: move[1] += speed

    # Compute camera axes
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
    current_time = pygame.time.get_ticks()
    if current_time - last_time >= fps_update_interval:
        fps = frame_count * 1000 / (current_time - last_time)
        frame_count = 0
        last_time = current_time

    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))

    # Switch to 2D rendering for FPS text
    glUseProgram(0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, game_width, game_height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Render FPS text as texture
    texture_id, tex_width, tex_height = render_text_to_texture(fps_text)
    draw_textured_quad(texture_id, tex_width, tex_height, -10, -10)
    glDeleteTextures(1, [texture_id])

    pygame.display.flip()
