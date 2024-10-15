"""
ロボットの関節角度と目標軌道のペアをデータセットとして集めるスクリプト  
集めたデータで教師あり学習するだけ
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import yaml
import argparse
import os
import pandas as pd

import mink

_ROOT=Path(__file__).parent.parent.parent
_XML = _ROOT / "model" / "scene.xml"

import sys
sys.path.append(str(_ROOT))
from src.utils import np2SE3_target
from src.trajectory_generator import generate_eight_trajectory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--configpath",required=True)
    args=parser.parse_args()

    # 軌道設定のYAMLファイルを読み込む
    config_path=args.configpath
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # >> 軌道生成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trajectory_config = config['trajectory']
    origin = tuple(trajectory_config['origin'])
    max_distance = tuple(trajectory_config['max_distance'])
    angle = trajectory_config['angle']
    num_loops = trajectory_config['num_loops']
    loop_duration = trajectory_config['loop_duration']
    delta_time = trajectory_config['delta_time']
    noise_std=trajectory_config["noise_std"]
    trajectory = generate_eight_trajectory(
        origin, max_distance, num_loops, 
        loop_duration, delta_time,angle,noise_std
    )

    _,target_x,target_y=trajectory.pop(0)
    T_wt=np2SE3_target( #初期座標
        quaternion=np.array([1,0,0,0]),
        position=np.array([target_x,target_y,0.3])
    )
    # << 軌道生成 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    result_path=Path(args.configpath).parent/"output"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

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


    datasets=[]
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


            if len(trajectory)<1: break #軌道がなくなったらおしまい


            # >>目標位置と姿勢を指定する>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            elapsed_t_from_set_target+=rate.dt
            if elapsed_t_from_set_target>delta_time: #制御周期と目標軌道周期は違う
                target_t,target_x,target_y=trajectory.pop(0)
                target_z=0.3 #z座標は固定
                T_wt=np2SE3_target(
                    quaternion=np.array([1,0,0,0]), #姿勢は固定
                    position=np.array([target_x,target_y,target_z])
                )
                elapsed_t_from_set_target=0.0 #経過時間をリセット
                print(T_wt)

                datasets.append(( #ロボットの状態と目標軌道を記録
                    target_t,
                    *data.qpos, #各関節の角度
                    target_x,target_y,target_z, #目標の手先位置
                ))
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


    datasets_db=pd.DataFrame(
        datasets,
        columns=[
            "time",
            *[f"joint{i}" for i in range(6)],
            "target_x","target_y","target_z"
        ]
    )
    datasets_db.to_csv(result_path/"datasets.csv",index=False)



if __name__ == "__main__":
    main()