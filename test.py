import os

# switch to "osmesa" or "egl" before loading pyrender
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt

# generate mesh
sphere = trimesh.creation.box(subdivision=4, extents=[1, 2, 3])
sphere.vertices += 1e-2 * np.random.randn(*sphere.vertices.shape)
mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)

# compose scene
scene = pyrender.Scene()

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
# light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0)
scene.add(mesh, pose=np.eye(4))

c = 0
scene.add(light, pose=np.eye(4))
scene.add(camera, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# render scene
flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
r = pyrender.OffscreenRenderer(512, 512)

color, _ = r.render(scene)

plt.figure(figsize=(8, 8))
plt.imshow(color)
plt.show()
