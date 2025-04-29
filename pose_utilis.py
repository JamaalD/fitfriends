import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_landmark_coords(landmarks):
    return {
        "left_shoulder": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
        },
        "right_shoulder": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
        },
        "left_elbow": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
        },
        "right_elbow": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
        },
        "left_wrist": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
        },
        "right_wrist": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
        },
        "left_hip": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
        },
        "right_hip": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
        },
        "left_knee": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
        },
        "right_knee": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
        },
        "left_ankle": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility,
        },
        "right_ankle": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility,
        },
        "left_heel": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility,
        },
        "right_heel": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility,
        },
        "left_foot_index": {
            "x": landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
            "y": landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
            "z": landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility,
        },
        "right_foot_index": {
            "x": landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
            "y": landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
            "z": landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z,
            "visibility": landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility,
        },
    }

def calculate_angle(ax, ay, bx, by, cx, cy):
    """Calculates the angle between three points (A, B, C)."""
    np_a = np.array([ax, ay])
    np_b = np.array([bx, by])
    np_c = np.array([cx, cy])
    
    ba = np_a - np_b
    bc = np_c - np_b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)

    return np_b, angle_degrees


def get_relevant_joints(workout, landmarks):
    """Returns the relevant joint coordinates for different workouts."""
    if workout == "squat" or "situp":
        return {
            "hip": calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ),
            "knee": calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            )
        }
    
    elif workout == "pushup":
        return {
            "elbow": calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ),
            "shoulder": calculate_angle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            )
        }
    
    else:
        return {}