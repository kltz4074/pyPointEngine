import numpy as np
import math


# ---------------- Matrix Utilities ----------------
def multiply_matrices(A, B):
    """Multiply two 4x4 matrices using NumPy."""
    return np.dot(np.array(A, dtype=np.float32), np.array(B, dtype=np.float32)).tolist()


def apply_matrix(v, M):
    """Apply 4x4 matrix to 4D vector using NumPy."""
    return np.dot(np.array(M, dtype=np.float32), np.array(v, dtype=np.float32)).tolist()


# ---------------- Core Classes ----------------
class Transform:
    def __init__(self, position=(0.0, 0.0, 0.0),
                 rotation=(0.0, 0.0, 0.0),
                 scale=(1.0, 1.0, 1.0)):
        self.position = position
        self.rotation = rotation
        self.scale = scale

    def get_matrix(self):
        px, py, pz = self.position
        rx, ry, rz = self.rotation
        sx, sy, sz = self.scale

        # Scale
        S = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Rotate X
        cx, sx_ = math.cos(rx), math.sin(rx)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx_, 0],
            [0, sx_, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Rotate Y
        cy, sy_ = math.cos(ry), math.sin(ry)
        Ry = np.array([
            [ cy, 0, sy_, 0],
            [ 0, 1, 0, 0],
            [-sy_, 0, cy, 0],
            [ 0, 0, 0, 1]
        ], dtype=np.float32)

        # Rotate Z
        cz, sz_ = math.cos(rz), math.sin(rz)
        Rz = np.array([
            [cz, -sz_, 0, 0],
            [sz_, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Translate
        T = np.array([
            [1, 0, 0, px],
            [0, 1, 0, py],
            [0, 0, 1, pz],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # M = T * Rz * Ry * Rx * S
        return np.dot(T, np.dot(Rz, np.dot(Ry, np.dot(Rx, S)))).tolist()


class GameObject:
    def __init__(self):
        self.transform = Transform()


class TriangleObject(GameObject):
    def __init__(self, vertices, indices, color=(1.0, 1.0, 1.0)):
        super().__init__()
        self.vertices = np.array(vertices, dtype=np.float32)   # For OpenGL
        self.indices = np.array(indices, dtype=np.uint32)      # Triangle indices
        self.color = np.array(color, dtype=np.float32) / 255.0 # Normalize to [0,1]

        self.vao = None  # Will be set up in OpenGL context
        self.vbo = None
        self.ebo = None

    def setup_buffers(self, gl):
        # Create and bind VAO
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)

        # Create and bind EBO
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)

        # Set vertex attributes (position)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4*4, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # Unbind
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)


class Camera(GameObject):
    def __init__(self, fov=70, aspect=1.0, near=0.1, far=100.0):
        super().__init__()
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def get_view_matrix(self):
        M = np.array(self.transform.get_matrix(), dtype=np.float32)
        R = M[0:3, 0:3].T  # Transpose of rotation/scale
        t = -np.dot(R, M[0:3, 3])

        view = np.eye(4, dtype=np.float32)
        view[0:3, 0:3] = R
        view[0:3, 3] = t

        return view.tolist()


# Load .obj with triangulation
def load_obj(path, scale=1.0, offset=(0, 0, 0)):
    verts, faces = [], []
    ox, oy, oz = offset

    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                x, y, z = map(float, line.split()[1:4])
                verts.append([x*scale+ox, y*scale+oy, z*scale+oz, 1])
            elif line.startswith('f '):
                idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(idx)-1):
                    faces.append((idx[0], idx[i], idx[i+1]))

    return verts, faces


# Projection matrix
def create_projection_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1
    return proj.tolist()
