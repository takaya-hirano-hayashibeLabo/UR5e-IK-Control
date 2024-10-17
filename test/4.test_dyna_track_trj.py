"""
dynamic forwadで軌道速度を変えてみる
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

import torch
import pandas as pd
from copy import deepcopy
sys.path.append(str(_ROOT.parent)) #自作ライブラリも入れたい
from DynamicSNN.src.model import DynamicSNN, ContinuousSNN, ThresholdEncoder
from DynamicSNN.src.utils import load_yaml,load_json2dict

if __name__ == "__main__":

    nn_modelpath=_ROOT.parent/"DynamicSNN/train-trajectory/output/20241017/dynasnn_eight_fig_ideal"
    trj_datapath=_ROOT/"main/collect_dataset/20241017/eight_figure_ideal/output/datasets.csv"

    nn_conf=load_yaml(nn_modelpath/"conf.yml")
    time_enc=DynamicSNN(conf=nn_conf["model"])
    time_enc.eval()

    # # time_encの最終層の重みを取得して表示
    # time_enc_weights = list(time_enc.parameters())[-1]
    # print("Time Encoder Final Layer Weights:")
    # print(time_enc_weights)

    nn_model=ContinuousSNN(
        nn_conf["output-model"],time_encoder=time_enc
    )

    weights=torch.load(
        nn_modelpath/"result/models/model_best.pth",
        map_location="cpu",
    )
    nn_model.load_state_dict(weights)
    nn_model.eval()

    sequence=50#nn_conf["train"]["sequence"]

    # # nn_modelの最終層の重みを取得して表示
    # nn_model_weights = list(nn_model.time_encoder.parameters())[-1]
    # print("SNN Model Final Layer Weights:")
    # print(nn_model_weights)


    #>> encoderの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    encoder_conf=nn_conf["encoder"]
    if encoder_conf["type"].casefold()=="thr":
        encoder=ThresholdEncoder(
            thr_max=encoder_conf["thr-max"],thr_min=encoder_conf["thr-min"],
            resolution=encoder_conf["resolution"]
        )
    else:
        raise ValueError(f"encoder type {encoder_conf['type']} is not supportated...")
    #<< encoderの準備 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    datasets=pd.read_csv(trj_datapath)
    
    input_labels=[f"endpos_{label}" for label in ["x","y"]]
    input_datas=datasets[input_labels]
    input_max=input_datas.max()
    input_max.name="max"
    input_min=input_datas.min()
    input_min.name="min"

    target_datas=datasets[["target_x","target_y"]]
    if nn_conf["output-model"]["out-type"].casefold()=="velocity":
        target_datas=target_datas.diff().iloc[1:] #最初の行はNaNになるので除外
    elif nn_conf["output-model"]["out-type"].casefold()=="position":
        target_datas=target_datas.iloc[1:]
    target_max=target_datas.max()
    target_max.name="max"
    target_min=target_datas.min()
    target_min.name="min"

    n_head=100#sequence
    in_trajectory=[]
    time_scales=[]
    #>> データの準備 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




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


    try:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            # Initialize to the home keyframe.
            configuration.update_from_keyframe("home")

            # Initialize the mocap target at the end-effector site.
            mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

            rate = RateLimiter(frequency=500.0, warn=False)
            run_count=0
            delta_time=0.07
            elapsed_time=delta_time
            timescale=1.4
            while viewer.is_running():


                #>> 目標軌道の推定 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                elapsed_time+=rate.dt
                if elapsed_time>delta_time:

                    elapsed_time=0.0 #経過時間のリセット

                    if run_count<n_head:
                        in_trajectory.append(input_datas.values[run_count])
                        time_scales.append(1.0)
                    else:                        
                        in_x=np.array(in_trajectory)[-sequence:] if len(in_trajectory)>sequence else np.array(in_trajectory)
                        in_x=2*(in_x-input_min.values)/(input_max.values-input_min.values)-1
                        in_spike=encoder(torch.Tensor(in_x).unsqueeze(0))

                        in_scales=np.array(time_scales)[-sequence:] if len(time_scales)>sequence else np.array(time_scales)

                        with torch.no_grad():
                            out_nrm=nn_model.dynamic_forward_given_scale(
                                in_spike.flatten(start_dim=2), torch.Tensor(in_scales)
                            )[0,-1].to("cpu").detach().numpy()
                        out=0.5*(out_nrm+1)*(target_max.values-target_min.values)+target_min.values
                        print(f"out nrm: {out_nrm}, out: {out}")

                        next_state=in_trajectory[-1]+out/timescale #差分を足し合わせる
                        in_trajectory.append(next_state)
                        time_scales.append(timescale)
                        # time_scales.append(1.0)
                        

                    target_x,target_y=in_trajectory[-1]
                    run_count+=1
                #<< 目標軌道の推定 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


                # Update task target.
                T_wt = np2SE3_target(
                    quaternion=[1,0,0,0],
                    position=[target_x,target_y,0.3]
                )
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

            
    finally:
        trajectory=pd.DataFrame(
            np.concatenate([np.array(in_trajectory),np.array(time_scales).reshape(-1,1)],axis=1),
            columns=["target_x","target_y","timescale"]
        )
        trajectory.to_csv(nn_modelpath/f"dynamic_trajectory_timescale{timescale:.2f}.csv")