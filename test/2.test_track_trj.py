"""
生成した軌道に沿ってロボットアームを動かすサンプル
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_ROOT=Path(__file__).parent.parent
_XML = _ROOT / "model" / "scene.xml"

import sys
sys.path.append(str(_ROOT))
from src.utils import np2SE3_target
from src.trajectory_generator import generate_eight_trajectory


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["wrist_3_link"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    mid = model.body("target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"


    # >> 軌道生成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    origin = (0.5, 0)
    max_distance = (0.3, 0.2)
    angle=np.pi/2.0
    num_loops = 5
    loop_duration = 10.0  # 1つの軌道を描く時間 [s]
    delta_time = 0.01 #目標軌道の更新周期
    noise_std=0.0005
    trajectory = generate_eight_trajectory(origin, max_distance, num_loops, loop_duration, delta_time,angle,noise_std)

    _,target_x,target_y=trajectory.pop(0)
    T_wt=np2SE3_target( #初期座標
        quaternion=np.array([1,0,0,0]),
        position=np.array([target_x,target_y,0.3])
    )
    # << 軌道生成 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        elapsed_t_from_set_target=0 #targetを指定してからの経過時間
        while viewer.is_running():


            # >>目標位置と姿勢を指定する>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            elapsed_t_from_set_target+=rate.dt
            if elapsed_t_from_set_target>delta_time: #制御周期と目標軌道周期は違う
                _,target_x,target_y=trajectory.pop(0)
                T_wt=np2SE3_target(
                    quaternion=np.array([1,0,0,0]),
                    position=np.array([target_x,target_y,0.3])
                )
                elapsed_t_from_set_target=0.0 #経過時間をリセット
                print(T_wt)
            # <<目標位置と姿勢を指定する<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            end_effector_task.set_target(T_wt) #ここでSE3型のクォータニオンとxyz座標を与えるだけ

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
