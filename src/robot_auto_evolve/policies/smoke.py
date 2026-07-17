from __future__ import annotations

import numpy as np

from robot_auto_evolve.agent import VLARequest
from robot_auto_evolve.protocol import (
    CameraObservation,
    FairObservation,
    RobotProprioception,
    RobotStateSpec,
    RobotStateVector,
)


TASKS = {
    "xvla_libero": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "xvla_calvin": "open_drawer",
    "xvla_simpler_widowx": "widowx_spoon_on_towel",
    "xvla_simpler_google_va": "google_robot_open_drawer",
    "xvla_simpler_google_vm": "google_robot_open_drawer",
    "openvla_simpler_google_va": "google_robot_open_drawer",
    "openvla_simpler_google_vm": "google_robot_open_drawer",
    "xvla_robotwin2": "open_laptop",
    "xvla_vlabench": "select_book",
    "pi05_libero": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "smolvla_robocerebra": "robocerebra_public60::Ideal::case1",
    "rlinf_pi05_libero": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "rlinf_pi05_libero_pro": "libero_pro_spatial_task::pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "molmoact2_libero": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "molmoact2_think_libero": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "molmoact2_droid": "BananaInBowlTableTask",
    "molmobot_droid": "BananaInBowlTableTask",
    "openpi_pi05_droid_jointpos": "BananaInBowlTableTask",
    "openpi_pi0_fast_droid_jointpos": "BananaInBowlTableTask",
}


CAMERAS = {
    "xvla_libero": ("main", "wrist"),
    "xvla_calvin": ("static", "wrist"),
    "xvla_simpler_widowx": ("main",),
    "xvla_simpler_google_va": ("main",),
    "xvla_simpler_google_vm": ("main",),
    "openvla_simpler_google_va": ("main",),
    "openvla_simpler_google_vm": ("main",),
    "xvla_robotwin2": ("head", "left_wrist", "right_wrist"),
    "xvla_vlabench": ("front", "main", "wrist"),
    "pi05_libero": ("main", "wrist"),
    "smolvla_robocerebra": ("main", "wrist"),
    "rlinf_pi05_libero": ("main", "wrist"),
    "rlinf_pi05_libero_pro": ("main", "wrist"),
    "molmoact2_libero": ("main", "wrist"),
    "molmoact2_think_libero": ("main", "wrist"),
    "molmoact2_droid": ("external", "wrist"),
    "molmobot_droid": ("external", "wrist"),
    "openpi_pi05_droid_jointpos": ("external", "wrist"),
    "openpi_pi0_fast_droid_jointpos": ("external", "wrist"),
}


def _pose(name: str, quaternion_order: str = "xyzw") -> RobotStateVector:
    return RobotStateVector(
        RobotStateSpec(
            name=name,
            quantity="end_effector_pose",
            frame_id=name,
            reference_frame="base",
            component_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
            units=("meter", "meter", "meter", "quaternion", "quaternion", "quaternion", "quaternion"),
            representation="xyz_quaternion",
            quaternion_order=quaternion_order,
        ),
        np.array(
            (0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0)
            if quaternion_order == "wxyz"
            else (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
            dtype=np.float32,
        ),
    )


def _gripper(name: str, width: int) -> RobotStateVector:
    return RobotStateVector(
        RobotStateSpec(
            name=name,
            quantity="gripper_position",
            frame_id=name,
            reference_frame="base",
            component_names=tuple(f"finger_{index}" for index in range(width)),
            units=("normalized",) * width,
            representation="vector",
            quaternion_order="none",
        ),
        np.zeros(width, dtype=np.float32),
    )


def _calvin_gripper() -> RobotStateVector:
    return RobotStateVector(
        RobotStateSpec(
            name="gripper_action",
            quantity="base_control_state",
            frame_id="calvin_gripper",
            reference_frame="calvin_robot_base",
            component_names=("open_positive",),
            units=("normalized",),
            representation="vector",
            quaternion_order="none",
        ),
        np.ones(1, dtype=np.float32),
    )


def synthetic_request(route_name: str, session_id: str = "startup-smoke") -> tuple[str, VLARequest]:
    task = TASKS[route_name]
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cameras = {
        name: CameraObservation(name, "opencv_rdf", image, None, None, None, None)
        for name in CAMERAS[route_name]
    }
    if route_name == "xvla_robotwin2":
        vectors = (_pose("left_eef_pose", "wxyz"), _gripper("left_gripper_position", 1), _pose("right_eef_pose", "wxyz"), _gripper("right_gripper_position", 1))
    elif route_name == "xvla_calvin":
        vectors = (_pose("eef_pose"), _calvin_gripper())
    elif route_name == "xvla_vlabench":
        vectors = (_pose("eef_pose", "wxyz"), _gripper("gripper_position", 1))
    elif route_name in {
        "molmoact2_droid",
        "molmobot_droid",
        "openpi_pi05_droid_jointpos",
        "openpi_pi0_fast_droid_jointpos",
    }:
        vectors = (
            RobotStateVector(
                RobotStateSpec(
                    name="arm_joint_position",
                    quantity="joint_position",
                    frame_id="franka_arm",
                    reference_frame="joint_space",
                    component_names=tuple(f"joint_{index}" for index in range(1, 8)),
                    units=("radian",) * 7,
                    representation="vector",
                    quaternion_order="none",
                ),
                np.zeros(7, dtype=np.float32),
            ),
            _gripper("gripper_position", 1),
        )
    else:
        width = 2 if route_name in {
            "xvla_libero",
            "pi05_libero",
            "smolvla_robocerebra",
            "rlinf_pi05_libero",
            "rlinf_pi05_libero_pro",
            "molmoact2_libero",
            "molmoact2_think_libero",
        } else 1
        vectors = (_pose("eef_pose"), _gripper("gripper_position", width))
    observation = FairObservation(
        episode_id="startup-smoke",
        step_index=0,
        timestamp_ns=0,
        instruction=task.replace("_", " "),
        cameras=cameras,
        proprioception=RobotProprioception(tuple(sorted(vectors, key=lambda item: item.spec.name))),
    )
    return task, VLARequest("startup-smoke-act", session_id, observation, observation.instruction, refresh=True)
