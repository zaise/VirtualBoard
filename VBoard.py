import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    list_points = []
    dHand = 'R' # Right: R / Left: L

    while True:
        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                cx, cy = calc_palm_moment(debug_image, hand_landmarks)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                debug_image, list_points = draw_landmarks(debug_image, cx, cy,
                                             hand_landmarks, handedness, list_points, dHand)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        debug_image = draw_blackboard(list_points, debug_image)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('MediaPipe Hand Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0: 
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1: 
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5: 
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, cx, cy, landmarks, handedness, list_points, dHand):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    if dHand == handedness.classification[0].label[0]:
        if not (landmarks.landmark[8].visibility < 0 or landmarks.landmark[8].presence < 0):
            in_x = landmarks.landmark[8].x
            in_y = landmarks.landmark[8].y
            pix_in_x = min(int(in_x * image_width), image_width - 1)
            pix_in_y = min(int(in_y * image_height), image_height - 1)
            cv.circle(image, (pix_in_x, pix_in_y), 20, (0, 0, 255), 2)
            cv.circle(image, (pix_in_x, pix_in_y), 5, (0, 255, 0), 2)
            cv.circle(image, (pix_in_x, pix_in_y), 12, (0, 255, 0), 2)
            landmark_point.append([pix_in_x, pix_in_y])

        if not (landmarks.landmark[12].visibility < 0 or landmarks.landmark[12].presence < 0):
            med_x = landmarks.landmark[12].x
            med_y = landmarks.landmark[12].y
            pix_med_x = min(int(med_x * image_width), image_width - 1)
            pix_med_y = min(int(med_y * image_height), image_height - 1)
            cv.circle(image, (pix_med_x, pix_med_y), 20, (0, 0, 255), 2)
            cv.circle(image, (pix_med_x, pix_med_y), 5, (0, 255, 0), 2)
            cv.circle(image, (pix_med_x, pix_med_y), 12, (0, 255, 0), 2)
            landmark_point.append([pix_med_x, pix_med_y])
        
        pix_in = np.asarray([pix_in_x, pix_in_y])
        pix_med = np.asarray([pix_med_x, pix_med_y])
        dis_finger = np.linalg.norm(pix_in - pix_med)
        if dis_finger < 15:
            list_points.append([pix_in_x, pix_in_y])
    
    else:
        if not (landmarks.landmark[8].visibility < 0 or landmarks.landmark[8].presence < 0):
            in_x = landmarks.landmark[8].x
            in_y = landmarks.landmark[8].y
            pix_in_x = min(int(in_x * image_width), image_width - 1)
            pix_in_y = min(int(in_y * image_height), image_height - 1)
            cv.circle(image, (pix_in_x, pix_in_y), 20, (0, 0, 255), 2)
            cv.circle(image, (pix_in_x, pix_in_y), 5, (0, 255, 0), 2)
            cv.circle(image, (pix_in_x, pix_in_y), 12, (0, 255, 0), 2)
            landmark_point.append([pix_in_x, pix_in_y])

        if not (landmarks.landmark[12].visibility < 0 or landmarks.landmark[12].presence < 0):
            med_x = landmarks.landmark[12].x
            med_y = landmarks.landmark[12].y
            pix_med_x = min(int(med_x * image_width), image_width - 1)
            pix_med_y = min(int(med_y * image_height), image_height - 1)
            cv.circle(image, (pix_med_x, pix_med_y), 20, (0, 0, 255), 2)
            cv.circle(image, (pix_med_x, pix_med_y), 5, (0, 255, 0), 2)
            cv.circle(image, (pix_med_x, pix_med_y), 12, (0, 255, 0), 2)
            landmark_point.append([pix_in_x, pix_in_y])
        
        pix_in = np.asarray([pix_in_x, pix_in_y])
        pix_med = np.asarray([pix_med_x, pix_med_y])
        dis_finger = np.linalg.norm(pix_in - pix_med)
        if dis_finger < 15:
            list_points = []

    if len(landmark_point) > 0:

        cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv.putText(image, handedness.classification[0].label[0],
                   (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                   2, cv.LINE_AA)

    return image, list_points


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image

def draw_blackboard(list_points, image):
    for point in list_points:
        cv.circle(image, tuple(point), radius=5, color=(255,0,0), thickness=-1)
    
    return image

def check_new_point(last_point, new_point, dis_radio):
    last = np.asarray(last_point)
    new = np.asarray(new_point)
    radio = np.linalg.norm(last-new)
    return radio < dis_radio

if __name__ == '__main__':
    main()
