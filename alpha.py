import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import json

app = FastAPI()
mp_pose = mp.solutions.pose

# Store client states (e.g., reps, phase, workout type)
client_state = {}

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


def analyse_frame(frame, pose, workout, phase, reps, last_feedback):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if not results.pose_landmarks:
        return "Make sure you're visible to the camera!", reps, phase

    landmarks = results.pose_landmarks.landmark
    angles = get_relevant_joints(workout, landmarks)
    landmark_coords = get_landmark_coords(landmarks)
    LKX = landmark_coords["left_knee"]["y"]
    LFX = landmark_coords["left_foot_index"]["y"]


    try:
        if workout == "squat":
            _, left_hip_angle = angles.get("hip", (None, 180))
            _, left_knee_angle = angles.get("knee", (None, 180))
            
            feedback = "squat with control"

            
            if phase == 1:
                if 130 < left_hip_angle < 160 and 130 < left_knee_angle < 160:
                    phase = 2  
                   
                    
            elif (LKX>=LFX*1.15):
                feedback = "your knees are too far forward"

            elif (left_hip_angle <= 30 and left_knee_angle <= 35):
                feedback = "your Squat is too deep."

            
            elif phase == 2:
                if left_hip_angle <= 120 and left_knee_angle <= 120:
                    phase = 4  
                    


            elif phase == 4 and (left_hip_angle <= 40 and left_knee_angle <= 75):
                phase = 3  
                feedback = "Great depth! Now start coming back up steadily."

            
            elif phase == 3 and left_hip_angle >= 120 and left_knee_angle >= 130:
                phase = 5 
                feedback = "nice squat! Push up with control and keep your back straight."

           
            elif phase == 4 and left_hip_angle >= 130 and left_knee_angle >= 130:
                phase = 1  
                feedback = "Make sure to go all the way down for a full squat!"


            elif phase == 5 and left_hip_angle >= 150 and left_knee_angle >= 150:
                phase = 1 
                reps += 1  
                

            else:
                feedback = last_feedback

        elif workout == "pushup":
            _, left_elbow_angle = angles.get("elbow", (None, 180))
            feedback = "make sure to control your movement"
            


            if phase == 1:
                if 140 < left_elbow_angle < 160:
                    phase = 2 
                 

            elif phase == 2 and left_elbow_angle <= 110:
                phase = 4  
    
                        
            elif phase == 4 and (left_elbow_angle <= 90):
                phase = 3  
                feedback = "Great form! Push back up with control."


            elif phase == 3 and left_elbow_angle >= 110:
                phase = 5  

    
            elif phase == 4 and left_elbow_angle >= 120:
                phase = 1 
                feedback = "Make sure to go all the way down for a full push-up!"
            

            elif phase == 5 and left_elbow_angle >= 150:
                phase = 1  
                reps += 1  
                feedback = "Great form!, keep your core tight."
                

            else:
                feedback = last_feedback
        
        elif workout == "situp":
            _, left_hip_angle = angles.get("hip", (None, 180))
            _, left_knee_angle = angles.get("knee", (None, 180))
            feedback = "make sure to not use momentum"


            
            if phase == 1:
                if 70 < left_hip_angle < 90:
                    phase = 2 
                
            elif left_knee_angle >= 110:
                feedback = "bend your knees more"
            
            elif phase == 2 and left_hip_angle <= 60:
                phase = 4 
    
                        
                        
            elif phase == 4 and (left_hip_angle <= 55):
                phase = 3  
                feedback = "Great form! Lower back down slowly."

                       

            elif phase == 3 and left_hip_angle >= 60:
                phase = 5 
                
              

            elif phase == 4 and left_hip_angle >= 70:
                phase = 1  
                feedback = "Make sure to go all the way up for a full sit-up!"
            

            elif phase == 5 and left_hip_angle >= 90:
                phase = 1 
                reps += 1  
                feedback = "Great form!, keep your core tight."

            else:
                feedback = last_feedback

    except Exception as e:
        feedback = f"Error processing landmarks: {str(e)}"

    return feedback, reps, phase


@app.websocket("/workout")
async def workout_websocket(websocket: WebSocket):
    await websocket.accept()
    print('connection established')
    last_feedback = ""
    
    client_state[websocket] = {
        "reps": 0,
        "phase": 1,
        "workout": None,  # Default workout, can be changed dynamically
    }

    try:
        while client_state[websocket]["workout"] is None:
            data = await websocket.receive_text()
            message = json.loads(data)
            if "workout" not in message:
                continue
            client_state[websocket]["workout"] = message["workout"]
            
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                try:
                    data = await websocket.receive_bytes()
                    info = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(info, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                    state = client_state.get(websocket, None)
                    if not state:
                        break

                    workout = state["workout"]
                    phase = state["phase"]
                    reps = state["reps"]

                    feedback, reps, phase = analyse_frame(frame, pose, workout, phase, reps, last_feedback)

                    client_state[websocket] = {"reps": reps, "phase": phase, "workout": workout}

                    await websocket.send_json({"reps": reps, "feedback": feedback})
                    last_feedback = feedback

                except Exception as e:
                    await websocket.send_json({"error": str(e)})
                    break

    except WebSocketDisconnect:
        # Client disconnected - CLEAN UP STATE
        del client_state[websocket]
    except Exception as e:
        # Other errors (don't delete state - client might still be connected)
        await websocket.send_json({"error": str(e)})
