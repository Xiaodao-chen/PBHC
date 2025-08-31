import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import torch
import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

# ========= 全局变量（供按键回调使用）=========
curr_start = 0
num_motions = 1
motion_id = 0
motion_acc = set()
time_step = 0.0
dt = 1/30
speed = 1.0
paused = False
rewind = False
motion_data_keys = []
contact_mask = None
curr_time = 0
resave = False

# 这些需要在回调里更新
motion_data = None
curr_motion_key = None
curr_motion = None
humanoid_fk = None
joint_gt = None
mj_model_global = None
vis_smpl_global = False
vis_tau_global = False
vis_tau_key_global = 'tau'
vis_contact_global = False


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
        np.zeros(3), np.zeros(9), rgba.astype(np.float32)
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        point1[0], point1[1], point1[2],
        point2[0], point2[1], point2[2]
    )


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind
    global motion_data_keys, contact_mask, curr_time, resave
    global motion_data, curr_motion_key, curr_motion, humanoid_fk, joint_gt, mj_model_global
    global vis_smpl_global, vis_tau_global, vis_tau_key_global, vis_contact_global

    try:
        ch = chr(keycode)
    except Exception:
        ch = ""

    if ch == "R":
        print("Reset")
        time_step = 0.0
        curr_time = 0

    elif ch == " ":
        print("Paused")
        paused = not paused

    elif keycode == 256 or ch == "Q":
        print("Esc")
        os._exit(0)

    elif ch == 'L':
        speed = speed * 1.5
        print("Speed: ", speed)

    elif ch == 'K':
        speed = speed / 1.5
        print("Speed: ", speed)

    elif ch == 'J':
        print("Toggle Rewind: ", not rewind)
        rewind = not rewind

    elif keycode == 262:  # Right
        time_step += dt

    elif keycode == 263:  # Left
        time_step -= dt

    elif ch == "Q":
        print('Modify left foot contact!!!')
        if contact_mask is not None:
            contact_mask[curr_time][0] = 1. - contact_mask[curr_time][0]
            resave = True

    elif ch == "E":
        print('Modify right foot contact!!!')
        if contact_mask is not None:
            contact_mask[curr_time][1] = 1. - contact_mask[curr_time][1]
            resave = True

    # ====== 新增：按 N 切换下一个动作 ======
    elif ch == "N":
        if num_motions <= 1:
            print("Only one motion available; cannot switch.")
            return

        # 切换索引（循环）
        motion_id = (motion_id + 1) % num_motions
        curr_motion_key = motion_data_keys[motion_id]
        curr_motion = motion_data[curr_motion_key]

        # 更新 dt / 模型时间步
        if 'fps' in curr_motion:
            dt = 1.0 / curr_motion['fps']
        # 如果模型已存在，更新其 timestep
        if mj_model_global is not None:
            mj_model_global.opt.timestep = dt

        # 更新接触掩码
        contact_mask = curr_motion['contact_mask'] if 'contact_mask' in curr_motion else None

        # 非 SMPL 可视化模式下，重新计算 FK 以得到新的关节可视化
        if not vis_smpl_global and humanoid_fk is not None:
            pose_aa = torch.from_numpy(curr_motion['pose_aa']).unsqueeze(0)
            root_trans = torch.from_numpy(curr_motion['root_trans_offset']).unsqueeze(0)
            with torch.no_grad():
                fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
                joint_gt = fk_return.global_translation_extend[0]

        # 复位时间
        time_step = 0.0
        curr_time = 0

        print(f"Switched to motion {motion_id + 1}/{num_motions}: {curr_motion_key} (dt={dt:.4f})")

    else:
        # 未映射按键
        if ch:
            print("not mapped", ch, keycode)
        else:
            print("not mapped keycode:", keycode)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind
    global motion_data_keys, contact_mask, curr_time, resave
    global motion_data, curr_motion_key, curr_motion, humanoid_fk, joint_gt, mj_model_global
    global vis_smpl_global, vis_tau_global, vis_tau_key_global, vis_contact_global

    curr_start, num_motions, motion_id, motion_acc, time_step, dt, speed, paused, rewind = 0, 1, 0, set(), 0, 1/30, 1.0, False, False

    motion_file = cfg.motion_file
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    num_motions = len(motion_data_keys)

    curr_motion_key = motion_data_keys[motion_id]
    curr_motion = motion_data[curr_motion_key]
    print(motion_file)

    speed = 1.0 if 'speed' not in cfg else cfg.speed
    hang = False if 'hang' not in cfg else cfg.hang
    if 'fps' in curr_motion:
        dt = 1.0 / curr_motion['fps']
    elif 'dt' in cfg:
        dt = cfg.dt

    print("Motion file: ", motion_file)
    print("Num motions: ", num_motions)
    print("Motion length (frames of first): ", motion_data[motion_data_keys[0]]['dof'].shape[0])
    print("Speed: ", speed)
    print()

    if 'contact_mask' in curr_motion.keys():
        contact_mask = curr_motion['contact_mask']
    else:
        contact_mask = None
    curr_time = 0
    resave = False

    humanoid_xml = "./description/robots/g1/g1_23dof_lock_wrist.xml"
    print(humanoid_xml)

    vis_smpl_global = False if 'vis_smpl' not in cfg else cfg.vis_smpl
    vis_tau_key_global = 'tau' if 'vis_tau_key' not in cfg else cfg.vis_tau_key
    vis_tau_global = (vis_tau_key_global in curr_motion) if 'vis_tau' not in cfg else cfg.vis_tau
    vis_contact_global = ('contact_mask' in curr_motion) if 'vis_contact' not in cfg else cfg.vis_contact

    if vis_smpl_global:
        assert 'smpl_joints' in curr_motion
    if vis_tau_global:
        assert vis_tau_key_global in curr_motion and not vis_contact_global
    if vis_contact_global:
        assert 'contact_mask' in curr_motion and not vis_tau_global

    # 非 SMPL 模式：初始化 FK 和 joint_gt
    if not vis_smpl_global:
        cfg_robot = OmegaConf.load("description/robots/g1/phc_g1_23dof.yaml")
        humanoid_fk = Humanoid_Batch(cfg_robot)  # load forward kinematics model
        pose_aa = torch.from_numpy(curr_motion['pose_aa']).unsqueeze(0)
        root_trans = torch.from_numpy(curr_motion['root_trans_offset']).unsqueeze(0)
        with torch.no_grad():
            fk_return = humanoid_fk.fk_batch(pose_aa, root_trans)
            joint_gt = fk_return.global_translation_extend[0]

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    mj_model_global = mj_model  # 给回调使用以动态更新 timestep

    print("Init Pose: ", (np.array(np.concatenate(
        [curr_motion['root_trans_offset'][0], curr_motion['root_rot'][0][[3, 0, 1, 2]], curr_motion['dof'][0]]
    ), dtype=np.float32)).__repr__())

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        viewer.cam.lookat[:] = np.array([0, 0, 0.7])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30

        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))

        while viewer.is_running():
            step_start = time.time()

            # 根据当前动作长度循环播放
            if time_step >= curr_motion['dof'].shape[0] * dt:
                time_step -= curr_motion['dof'].shape[0] * dt
            curr_time = round(time_step / dt) % curr_motion['dof'].shape[0]

            # 写入 qpos
            if hang:
                mj_data.qpos[:3] = np.array([0, 0, 0.8])
            else:
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]  # xyzw -> wxyz
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += dt * (1 if not rewind else -1) * speed

            # 可视化 SMPL 或机器人 FK 关节
            if vis_smpl_global:
                joint_gt_local = motion_data[curr_motion_key]['smpl_joints']
                if not np.all(joint_gt_local[curr_time] == 0):
                    for i in range(joint_gt_local.shape[1]):
                        viewer.user_scn.geoms[i].pos = joint_gt_local[curr_time, i]
            else:
                if joint_gt is not None:
                    for i in range(23):
                        viewer.user_scn.geoms[i + 1].pos = joint_gt[curr_time, i + 1]

            # 可视化接触或力矩（互斥）
            if vis_contact_global and contact_mask is not None:
                viewer.user_scn.geoms[6].rgba = np.array([0, 1 - curr_motion['contact_mask'][curr_time, 0], 0, 1])
                viewer.user_scn.geoms[12].rgba = np.array([0, 1 - curr_motion['contact_mask'][curr_time, 1], 0, 1])

            if vis_tau_global:
                scale_factor = 0.1
                for i in range(23):
                    tau = curr_motion[vis_tau_key_global][curr_time, i]
                    color_gradient = abs(tau) * scale_factor
                    if tau > 0:
                        viewer.user_scn.geoms[i + 1].rgba = np.array([0.8, 0.1, 0.1, 0.1 + color_gradient])
                    elif tau < 0:
                        viewer.user_scn.geoms[i + 1].rgba = np.array([0.1, 0.8, 0.1, 0.1 + color_gradient])

            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            print("Frame ID: ", curr_time, '\t | Time ', f"{time_step:4f}", end='\r\b')

    # 保存编辑后的接触（如有改动）
    if resave:
        motion_data[curr_motion_key]['contact_mask'] = contact_mask
        motion_file = motion_file.split('.')[0] + '_edit_cont.pkl'
        print(motion_file)
        joblib.dump(motion_data, motion_file)


if __name__ == "__main__":
    main()