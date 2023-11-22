import mediapipe as mp
import cv2 as cv
import datetime
import json
import sys
import os
import glob
import mimetypes
from argparse import ArgumentParser

from utility import draw_landmarks_on_image, draw_iris_circle, draw_eye_contour_points, check_eye_status, get_eye_contour_pos

model_path = 'face_landmarker_v2_with_blendshapes.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=10)

def find_landmarks(landmarker, frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
    # Perform face landmarking on the provided single image.
    face_landmarker_result = landmarker.detect(mp_image)
    
    # Draw landmarks on the image.
    result = draw_landmarks_on_image(frame, face_landmarker_result)
    
    regular_count = 0
    blink_count = 0
    cross_count = 0
    other_count = 0
    
    student_status = []
    
    for face_landmark in face_landmarker_result.face_landmarks:         
        # Draw circles at the iris centers
        result = draw_iris_circle(face_landmark, result)
        result = draw_eye_contour_points(face_landmark, result)
        
        eye_status = check_eye_status(face_landmark, frame)
        right_eye_contour_pos, left_eye_contour_pos = get_eye_contour_pos(face_landmark, frame)
        
        each_status = {
            "EyeStatus": eye_status,
            "EyePoints": {
                "RightEye": right_eye_contour_pos,
                "LeftEye": left_eye_contour_pos
            }
        }
        
        student_status.append(each_status)
        
        if eye_status == 'Regular':
            regular_count += 1
            continue
        if eye_status == 'Blink':
            blink_count += 1
            continue
        if eye_status == 'CrossEye':
            cross_count += 1
            continue
        if eye_status == 'Other':
            other_count += 1
            continue
    
    json_data = {
        "TimeFrame": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "GeneralStatus": {
            "Regular": regular_count,
            "Blink": blink_count,
            "CrossEye": cross_count,
            "Other": other_count
        },
        "StudentStatus": student_status
    }
    cv.FONT_HERSHEY_PLAIN
    cv.putText(result, f'Students: {len(student_status)}', (10 , 60), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    cv.putText(result, f'Regular: {regular_count}', (10 , 120), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv.putText(result, f'Blink: {blink_count}', (10, 180), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv.putText(result, f'CrossEye: {cross_count}', (10, 240), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv.putText(result, f'Other: {other_count}', (10, 300), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    
    return json_data, result


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, help="'Camera'  or 'Video' or 'Photo'", default="Camera")
    parser.add_argument('--video', type=str, help="Path to video file", default=None)
    parser.add_argument('--photo', type=str, help="Path to photo file", default=None)
    parser.add_argument('--output', type=str, help="Path to output file", default=None)
    
    args = parser.parse_args()
    
    source = args.source
    video = args.video
    photo = args.photo
    output = args.output
    
    landmarker = FaceLandmarker.create_from_options(options)
    
    if source == "Camera" or source == "Video":
        video_source = 0
        output_path = output
        if source == "Camera":
            video_source == 0
        if source == "Video":
            if video == None:
                print("You have to set Video path when you choose Video mode.")
                sys.exit(1)
            if os.path.isdir(video):
                print("Please enter file path, not directory")
                sys.exit(1)
            if 'video' not in mimetypes.guess_type(video)[0]:
                print("This file is not a video file.")
                sys.exit(1)
            video_source = video
        # Use OpenCV’s VideoCapture to load the input video.
        capture = cv.VideoCapture(video_source)
        
        # Get the frame size from the input video
        frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        
        # Get the video codec and create a VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or use other codecs like 'XVID'
        output_video = cv.VideoWriter(output if output != None else "output.mp4", fourcc, 15.0, frame_size)
        
        total_status = []
        cnt = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                print("Can't read video stream.")
                break
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            
            if cnt % 3 == 0:
                json_data, result_img = find_landmarks(landmarker, frame)
                
                total_status.append(json_data)
                    
                output_video.write(result_img)
                cv.imshow('frame', result_img)
                
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            cnt += 1
            
        capture.release()
        output_video.release()
        with open("output.json", "w") as file:
            json.dump(total_status, file, indent=4)
        cv.destroyAllWindows()

    if source == "Photo":
        photo_path = photo
        if photo_path == None:
            print("You must input '--photo' path when you choose 'Photo' mode.")
            sys.exit(1)
        if os.path.isdir(photo_path):
            image_paths = glob.glob(photo_path + "/*")  # get all the paths to the images
            output_path = output if output != None and os.path.isdir(output) else "output"
            os.makedirs(output_path, exist_ok=True)
            for image_path in image_paths:
                if 'image/' in mimetypes.guess_type(image_path)[0]:
                    print(image_path)
                    frame = cv.imread(image_path)
                    json_data, result_img = find_landmarks(landmarker, frame)
                    _, file_name = os.path.split(image_path)
                    print(f"Saving result to {output_path}/{file_name}")
                    json_file_name = file_name.split(".")[0] + ".json"
                    with open(f"{output_path}/{json_file_name}", "w") as file:
                        print(f"Saving result to {output_path}/{json_file_name}")
                        json.dump(json_data, file)
        else:
            if 'image' not in mimetypes.guess_type(photo_path)[0]:
                print("This file is not an image file.")
                sys.exit(1)
            frame = cv.imread(photo_path)
            json_data, result_img = find_landmarks(landmarker, frame)
            output_path = output if output != None else "output.jpg"
            cv.imwrite(output_path, result_img)
            print(f"Result Image is saved to {output_path}")
            json_file_name = output_path.split(".")[0] + ".json"
            with open(json_file_name, "w") as file:
                json.dump(json_data, file)

    landmarker.close()
    