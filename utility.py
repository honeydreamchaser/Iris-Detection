import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv

# plus Enum Defect { Regular = 0, Blink = 1, CrossEye = 2, Other = 3}
#
#            1 RightEyeUpOut(160)          2 RightEyeUpIn(158)                                   1 LeftEyeUpIn(385)         2 LeftEyeUpOut(387)
#
#  6 RightEyeOut(33)        0 RightEyePupil(468)          3 RightEyeIn(133)           6 LeftEyeIn(463)          0 LeftEyePupil(473)           3 LeftEyeOut(263)
#
#            5 RightEyeDownOut(144)        4 RightEyeDownIn(153)                                5 LeftEyeDownIn(380)       4 LeftEyeDownOut(373)
#
right_eye_contour_points = {
    "RightEyePupil": 468,
    "RightEyeUpOut": 160,
    "RightEyeUpIn": 158,
    "RightEyeIn": 133,
    "RightEyeDownIn": 153,
    "RightEyeDownOut": 144,
    "RightEyeOut": 33,
    "RightEyeRightPupil": 471,
    "RightEyeLeftPupil": 469
}

left_eye_contour_points = {
    "LeftEyePupil": 473,
    "LeftEyeUpIn": 385,
    "LeftEyeUpOut": 387,
    "LeftEyeOut": 263,
    "LeftEyeDownOut": 373,
    "LeftEyeDownIn": 380,
    "LeftEyeIn": 463,
    "LeftEyeRightPupil": 476,
    "LeftEyeLeftPupil": 474
}

def draw_iris_circle(face_landmark, frame):
    result = frame.copy()
    right_eye_center = (int(face_landmark[right_eye_contour_points["RightEyePupil"]].x * frame.shape[1]),
                    int(face_landmark[right_eye_contour_points["RightEyePupil"]].y * frame.shape[0]))
    left_eye_center = (int(face_landmark[left_eye_contour_points["LeftEyePupil"]].x * frame.shape[1]),
                        int(face_landmark[left_eye_contour_points["LeftEyePupil"]].y * frame.shape[0]))
    
    point2 = (int(face_landmark[469].x * frame.shape[1]),
        int(face_landmark[469].y * frame.shape[0]))
    point1 = (int(face_landmark[474].x * frame.shape[1]),
        int(face_landmark[474].y * frame.shape[0]))

    radius_right = int(math.sqrt((left_eye_center[0] - point1[0])**2 +(left_eye_center[1] - point1[1])**2))
    radius_left = int(math.sqrt((right_eye_center[0] - point2[0])**2 +(right_eye_center[1] - point2[1])**2))
    
    # Draw circles at the iris centers
    cv.circle(result, left_eye_center, radius=radius_left, color=(0, 255, 0), thickness=1)
    cv.circle(result, right_eye_center, radius=radius_right, color=(0, 0, 255), thickness=1)
    
    return result

def draw_eye_contour_points(face_landmark, frame):
    result = frame.copy()
    
    for key, val in right_eye_contour_points.items():
        point = (int(face_landmark[val].x * frame.shape[1]),
                int(face_landmark[val].y * frame.shape[0]))
        cv.circle(result, point, radius=2, color= (0, 0, 255), thickness=-1)
        capitalized_letters = [char for char in key if char.isupper()]
        capitalized_letters_string = ''.join(capitalized_letters)
        cx, cy = 0, 0
        if "Up" in key:
            cy = -5
        if "Down" in key:
            cy = 5
        if "Out" in key:
            text_size = cv.getTextSize(capitalized_letters_string, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cx = 0 - text_size[0][0] - 5
        if "In" in key:
            cx = 5
        point = (point[0] + cx, point[1] + cy)
        # cv.putText(result, capitalized_letters_string, point, cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    for key, val in left_eye_contour_points.items():
        point = (int(face_landmark[val].x * frame.shape[1]),
                int(face_landmark[val].y * frame.shape[0]))
        cv.circle(result, point, radius=2, color= (0, 255, 0), thickness=-1)
        capitalized_letters = [char for char in key if char.isupper()]
        capitalized_letters_string = ''.join(capitalized_letters)
        cx, cy = 0, 0
        if "Up" in key:
            cy = -5
        if "Down" in key:
            cy = 5
        if "Out" in key:
            cx = 5
        if "In" in key:
            text_size = cv.getTextSize(capitalized_letters_string, cv.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cx = 0 - text_size[0][0] - 5
        point = (point[0] + cx, point[1] + cy)
        # cv.putText(result, capitalized_letters_string, point, cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    return result

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0),
                                                                                thickness=1,
                                                                                circle_radius=1)
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255),
                                                                                thickness=1,
                                                                                circle_radius=1)
        )
    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()

def calculate_ear(eye_landmarks):
    eye_landmarks = [val for _, val in eye_landmarks.items()]
    horizontal_dist1 = math.sqrt((eye_landmarks[1][0] - eye_landmarks[5][0]) ** 2 + (eye_landmarks[1][1] - eye_landmarks[5][1]) ** 2)
    horizontal_dist2 = math.sqrt((eye_landmarks[2][0] - eye_landmarks[4][0]) ** 2 + (eye_landmarks[2][1] - eye_landmarks[4][1]) ** 2)
    vertical_dist = math.sqrt((eye_landmarks[3][0] - eye_landmarks[6][0]) ** 2 + (eye_landmarks[3][1] - eye_landmarks[6][1]) ** 2)

    # Calculate the EAR
    ear = (horizontal_dist1 + horizontal_dist2) / (2 * vertical_dist)
    return ear

def calculate_polygon_contain(eye_landmarks):
    eye_landmarks = [val for _, val in eye_landmarks.items()]
    cx1 = (eye_landmarks[2][0] - eye_landmarks[1][0]) // 3
    cx2 = (eye_landmarks[4][0] - eye_landmarks[5][0]) // 3
    x1, y1 = eye_landmarks[1][0] + cx1, eye_landmarks[1][1]
    x2, y2 = eye_landmarks[2][0] - cx1, eye_landmarks[2][1]
    x3, y3 = eye_landmarks[4][0] - cx2, eye_landmarks[4][1]
    x4, y4 = eye_landmarks[5][0] + cx2, eye_landmarks[5][1]
    eye_center_area = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    eye_center_point = eye_landmarks[0]
    
    result = cv.pointPolygonTest(np.array(eye_center_area), eye_center_point, False) 
    if result <= 0:
        return True
    else:
        return False

def check_is_blink(right_eye_contour_pos, left_eye_contour_pos):
    right_ear = calculate_ear(right_eye_contour_pos) 
    left_ear = calculate_ear(left_eye_contour_pos)
    
    if right_ear < 0.25 and left_ear < 0.25:
        return True
    else:
        return False

def check_is_cross_eye(right_eye_contour_pos, left_eye_contour_pos):
    right_is_center = calculate_polygon_contain(right_eye_contour_pos)
    left_is_center = calculate_polygon_contain(left_eye_contour_pos)
    if right_is_center or left_is_center:
        return True
    else:
        return False
    
def get_eye_contour_pos(face_landmark, frame):
    right_eye_contour_pos = {}
    left_eye_contour_pos = {}
    for key, val in right_eye_contour_points.items():
        point = (int(face_landmark[val].x * frame.shape[1]),
                int(face_landmark[val].y * frame.shape[0]))
        right_eye_contour_pos[key] = point
    
    for key, val in left_eye_contour_points.items():
        point = (int(face_landmark[val].x * frame.shape[1]),
                int(face_landmark[val].y * frame.shape[0]))
        left_eye_contour_pos[key] = point
    return right_eye_contour_pos, left_eye_contour_pos

def check_eye_status(face_landmark, frame):
    right_eye_contour_pos, left_eye_contour_pos = get_eye_contour_pos(face_landmark, frame)

    is_blink = check_is_blink(right_eye_contour_pos, left_eye_contour_pos)
    is_cross_eye = check_is_cross_eye(right_eye_contour_pos, left_eye_contour_pos)
    if is_blink:
        return "Blink"
    elif is_cross_eye:
        return "CrossEye"
    elif not is_blink and not is_cross_eye:
        return "Regular"
    return "Other"
