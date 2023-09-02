import os
import urllib.request
import time
import numpy as np
# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#get_started
class MediapipeHandLandmark():
    # https://storage.googleapis.com/mediapipe-models/
    base_url = 'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/'
    model_name = 'hand_landmarker.task'
    model_folder_path = './models'

    # visualize param
    H_MARGIN = 10  # pixels
    V_MARGIN = 30  # pixels
    RADIUS_SIZE = 3  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    RIGHT_HAND_COLOR = (0, 255, 0)
    LEFT_HAND_COLOR = (100, 100, 255)
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    # hand landmark id
    NUM_LMK = 21
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_FINGER_MCP = 17
    PINKY_FINGER_PIP = 18
    PINKY_FINGER_DIP = 19
    PINKY_FINGER_TIP = 20

    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#get_started
    def __init__(
            self,
            model_folder_path=model_folder_path,
            base_url=base_url,
            model_name=model_name,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
            ):

        model_path = self.set_model(base_url, model_folder_path, model_name)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.num_landmarks = self.NUM_LMK # default

    def set_model(self, base_url, model_folder_path, model_name):
        model_path = model_folder_path+'/'+model_name
        # download model if model file does not exist
        if not os.path.exists(model_path):
            # make directory if model_folder directory does not exist
            if not os.path.exists(model_folder_path):
                os.mkdir(model_folder_path)
            # download model file
            url = base_url+model_name
            save_name = model_path
            urllib.request.urlretrieve(url, save_name)
        return model_path

    def detect(self, img):
        self.size = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
        self.num_detected_hands = len(self.results.hand_landmarks)
        return self.results

    def get_normalized_landmark(self, id_hand, id_landmark):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        x = self.results.hand_landmarks[id_hand][id_landmark].x
        y = self.results.hand_landmarks[id_hand][id_landmark].y
        z = self.results.hand_landmarks[id_hand][id_landmark].z
        return np.array([x, y, z])

    def get_landmark(self, id_hand, id_landmark):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        height, width = self.size[:2]
        x = self.results.hand_landmarks[id_hand][id_landmark].x
        y = self.results.hand_landmarks[id_hand][id_landmark].y
        z = self.results.hand_landmarks[id_hand][id_landmark].z
        return np.array([int(x*width), int(y*height), int(z*width)])

    def get_landmark_presence(self, id_hand, id_landmark):
        return self.results.hand_landmarks[id_hand][id_landmark].presence

    def get_landmark_visibility(self, id_hand, id_landmark):
        return self.results.hand_landmarks[id_hand][id_landmark].visibility

    def get_handedness(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.results.handedness[id_hand][0].category_name

    def get_score_handedness(self, id_hand):
        if self.num_detected_hands == 0:
            print('no hand')
            return None
        return self.results.handedness[id_hand][0].score

    def visualize(self, image):
        annotated_image = np.copy(image)
        # for hand, info in zip(self.results.hand_landmarks, self.results.handedness):
        for i, hand in enumerate(self.results.hand_landmarks):
            handedness = self.get_handedness(i)
            score = self.get_score_handedness(i)
            wrist_point = self.get_landmark(i, 0)

            if self.get_handedness(i) == 'Right':
                color = self.RIGHT_HAND_COLOR
            else:
                color = self.LEFT_HAND_COLOR

            for j in range(len(hand)):
                point = self.get_landmark(i, j)
                cv2.circle(annotated_image, tuple(point[:2]), self.RADIUS_SIZE, color, thickness=self.FONT_THICKNESS)

            txt = handedness+'('+'{:#.2f}'.format(score)+')'
            wrist_point_for_text = (wrist_point[0]+self.H_MARGIN, wrist_point[1]+self.V_MARGIN)
            cv2.putText(annotated_image, org=wrist_point_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.FONT_SIZE, color=color, thickness=self.FONT_THICKNESS, lineType=cv2.LINE_4)
        return annotated_image

    def visualize_with_mp(self, image):
        hand_landmarks_list = self.results.hand_landmarks
        handedness_list = self.results.handedness
        annotated_image = np.copy(image)
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.V_MARGIN
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv2.LINE_AA)
        return annotated_image

    def release(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)
    Hand = MediapipeHandLandmark()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            print("Ignoring empty camera frame.")
            break

        # HandLandmarker requires horizontal flip for input image
        flipped_frame = cv2.flip(frame, 1)

        results = Hand.detect(flipped_frame)

        if Hand.num_detected_hands > 0:
            index_hand = 0 #
            index_landmark = Hand.WRIST #
            print(
                Hand.get_handedness(index_hand),
                'score:{:#.2f}'.format(Hand.get_score_handedness(index_hand)),
                'Wrist:',
                Hand.get_normalized_landmark(index_hand, index_landmark),
                Hand.get_landmark(index_hand, index_landmark)
                )

        annotated_image = Hand.visualize(flipped_frame)
        # annotated_image = Hand.visualize_with_mp(flipped_frame)

        cv2.imshow('annotated image', annotated_image)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__=='__main__':
    main()
