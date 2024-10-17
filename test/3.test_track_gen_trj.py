"""
NNが生成した目標軌道をトラックするテスト
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


    nn_modelpath=_ROOT.parent/"DynamicSNN/train-trajectory/output/20241017/dynasnn_ellipse_ideal"
    nn_conf=load_yaml(nn_modelpath/"conf.yml")
    time_enc=DynamicSNN(conf=nn_conf["model"])
    # time_encの最終層の重みを取得して表示
    time_enc_weights = list(time_enc.parameters())[-1]
    print("Time Encoder Final Layer Weights:")
    print(time_enc_weights)

    nn_model=ContinuousSNN(
        nn_conf["output-model"],time_encoder=time_enc
    )

    weights=torch.load(
        nn_modelpath/"result/models/model_best.pth",
        map_location="cpu",
    )
    nn_model.load_state_dict(weights)
        
    nn_model.eval()


    # nn_modelの最終層の重みを取得して表示
    nn_model_weights = list(nn_model.time_encoder.parameters())[-1]
    print("SNN Model Final Layer Weights:")
    print(nn_model_weights)


    sequence=100#nn_conf["train"]["sequence"]
    print(nn_model)

    thr_encoder=ThresholdEncoder(
        thr_max=nn_conf["encoder"]["thr-max"],
        thr_min=nn_conf["encoder"]["thr-min"],
        resolution=nn_conf["encoder"]["resolution"]
    )

    nn_out_type=nn_conf["output-model"]["out-type"]

    input_nrm_params_js :dict=load_json2dict(nn_modelpath/"result/input_nrm_params.json")
    output_nrm_params_js:dict=load_json2dict(nn_modelpath/"result/target_nrm_params.json")

    in_nrm_params={}
    for key,joints in input_nrm_params_js.items():
        print(joints.keys())
        in_nrm_params[key]=np.array(list(joints.values()))

    out_nrm_params={}
    for key, targets in output_nrm_params_js.items():
        print(targets.keys())
        out_nrm_params[key]=np.array(list(targets.values()))

    print(in_nrm_params)
    print(out_nrm_params)


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
    delta_time=0.07
    joint_trajectory=[]
    target_trajectory=[]

    base_trjpath=_ROOT/"main/collect_dataset/20241017/ellipse_ideal/output/datasets.csv"
    n_head=sequence
    base_trj=pd.read_csv(base_trjpath)[["target_x","target_y"]].values[:n_head+1] #最初の補助軌道
    next_target= base_trj[0]#軌道の最初の目標
    T_wt=np2SE3_target(
        quaternion=[1,0,0,0],
        position=[*next_target,0.3]
    )

    # << 軌道生成 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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
            elapsed_t_from_set_target=0 #targetを指定してからの経過時間
            target_count=0
            while viewer.is_running():

                # >>目標位置と姿勢を指定する>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                elapsed_t_from_set_target+=rate.dt
                if elapsed_t_from_set_target>delta_time: #制御周期と目標軌道周期は違う
                    elapsed_t_from_set_target=0.0 #経過時間をリセット
                    # joint_trajectory.append(list(data.qpos))
                    # joint_trajectory.append(list(data.site("attachment_site").xpos)[:-1])
                    joint_trajectory.append(next_target)
                    # if len(joint_trajectory)>sequence: 
                    #     joint_trajectory.pop(0) #捨てる
                    # print(len(joint_trajectory))

                    if target_count<n_head:
                        next_target=0.0+base_trj[target_count+1]

                    elif target_count>=n_head:
                        if target_count==n_head: print("***start prediction***")
                        in_x=np.array(joint_trajectory)[-sequence:] if len(joint_trajectory)>sequence else np.array(joint_trajectory)
                        
                        # print("======"*20)
                        # print(np.array(joint_trajectory))
                        # print("--")
                        # print(in_x)

                        in_x=2*((in_x-in_nrm_params["min"])/(in_nrm_params["max"]-in_nrm_params["min"]))-1 #正規化
                        
                        in_spike=thr_encoder(torch.Tensor(in_x).unsqueeze(0)) #[1 x T x xdim]
                        # print(f"in spike shape: {in_spike.shape}")

                        with torch.no_grad():
                            target_nrm=nn_model.forward(in_spike.flatten(start_dim=2))[0,-1].to("cpu").detach().numpy() #一番最後の出力だけ使う
                        target=0.5*(target_nrm+1)*(out_nrm_params["max"]-out_nrm_params["min"])+out_nrm_params["min"] #正規化解除
                        print("nrm target: ",target_nrm, "target: ",target)

                        if nn_out_type.casefold()=="position":
                            next_target=target
                        elif nn_out_type.casefold()=="velocity":
                            next_target=deepcopy(next_target+target)
                            # next_target=np.array(data.site("attachment_site").xpos[:-1])+target #差分を足す

                    target_count+=1

                    # print(f"delta target shape: {dtarget.shape}")
                    print(f"target count:{target_count}, endpos: {list(data.site('attachment_site').xpos)[:-1]}, target: {next_target}")

                    target_trajectory.append(list(next_target))
                    T_wt=np2SE3_target(
                        quaternion=[1,0,0,0],
                        position=[*next_target,0.3]
                    )
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
        
    finally:

        # joint_trajectory_db=pd.DataFrame(
        #     np.concatenate([joint_trajectory,target_trajectory],axis=1),
        #     columns=[*[f"joint{i}" for i in range(6)],*["target_x","target_y"]]
        # )
        joint_trajectory_db=pd.DataFrame(
            np.concatenate([joint_trajectory,target_trajectory],axis=1),
            columns=["endpos_x","endpos_y","target_x","target_y"]
        )
        joint_trajectory_db.to_csv(nn_modelpath/"joint_trajectory.csv",index=True)
