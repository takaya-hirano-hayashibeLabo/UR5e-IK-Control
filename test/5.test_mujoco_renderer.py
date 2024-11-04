"""
瀬戸が作ったmujoco上に線や球を描画するコード
"""

import mujoco
import numpy as np

import mujoco.viewer
import time
import cv2

def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_connector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                       point1, point2)

def add_visual_sphere(scene, point1, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_connector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[radius, 0, 0],
                      pos=point1,
                      mat=np.eye(3).flatten(),
                      rgba=rgba
                      )
  
  
def add_visual_line(scene, point1, point2, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        size=[1, 1, 1],
                        pos=np.array([0.5,0,0.2]),
                        mat=np.eye(3).flatten(),
                        rgba=rgba)
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_LINE, width,
                           point1[0], point1[1], point1[2], point2[0], point2[1], point2[2])
  
def modify_scene(scn):
    # rgba=np.array([1,0,0,1])
    # radius=.003
    # point1 = np.array([0,0,0])
    # point2 = np.array([0,0,0.3])
    # add_visual_capsule(scn, point1, point2, radius, rgba)
    
    # 線を描画
    rgba=np.array([0,0,1,1])
    width=2
    n_points = 100
    for i in range(n_points):
        point1 = np.array([np.sin(2*i*np.pi/n_points),np.cos(2*i*np.pi/n_points),0.3])
        point2 = np.array([np.sin(2*(i+1)*np.pi/n_points),np.cos(2*(i+1)*np.pi/n_points),0.3])
        add_visual_line(scn, point1, point2, width, rgba)

    # 球を描画
    point1 = np.array([0,0,0.3])
    rgba = np.array([0,1,0,1])
    radius = 0.1
    add_visual_sphere(scn, point1, radius, rgba)


xml = """
<mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>-->
            <!-- <texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>-->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.95 0.95 0.95" rgb2="0.99 0.99 0.99"  width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <camera name="cam1" pos="1.078 -2.235 0.990" xyaxes="0.901 0.434 -0.000 -0.161 0.334 0.929"/>

            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="0.8" rgba="1.0 1.0 1.0 1"/>
        </worldbody>

    </mujoco>
    """
    

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)



# viewerの場合
# ビューアを起動
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    #   viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.user_scn.ngeom = 0
        
        
        modify_scene(viewer.user_scn)
        
        viewer.sync()
        time.sleep(0.001)
        


# # Rendererの場合
# # レンダラーを初期化
renderer = mujoco.Renderer(model,1080,1920)

for i in range(500):

    
    mujoco.mj_step(model, data)
    renderer.update_scene(data,"cam1")
    modify_scene(renderer.scene)
    frame = renderer.render()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame_rgb)
    cv2.waitKey(1)  # Add this line to process window events
    time.sleep(0.001)