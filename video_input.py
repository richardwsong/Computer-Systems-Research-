import cv2
import mediapipe as mp
import numpy as np
import os

from imageio.plugins import ffmpeg

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Setup Time Series
leftShoulderX, leftShoulderY, rightShoulderX, rightShoulderY, leftElbowX, leftElbowY, rightElbowX, rightElbowY = [], [], [], [], [], [], [], []

count = 0

# For static images:
with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
    input_vid = cv2.VideoCapture('media/jump1.mp4')
    ret = 1
    while True:
        ret, frame = input_vid.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            continue

        # Render
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        cv2.imwrite('outputs/images/image' + str(count+10) + '.png', image)

        if count % 10 == 0:
            leftShoulderX.append([count, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x])
            leftShoulderY.append([count, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])

            rightShoulderX.append([count, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x])
            rightShoulderY.append([count, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

            leftElbowX.append([count, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x])
            leftElbowY.append([count, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])

            rightElbowX.append([count, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x])
            rightElbowY.append([count, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

        count += 1

    image_folder = 'outputs/images'
    video_name = 'outputs/vid_output.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    rightElbowX = np.array(rightElbowX)
    rightElbowY = np.array(rightElbowY)
    leftElbowX = np.array(leftElbowX)
    leftElbowY = np.array(leftElbowY)
    rightShoulderX = np.array(rightShoulderX)
    rightShoulderY = np.array(rightShoulderY)
    leftShoulderX = np.array(leftShoulderY)
    leftShoulderY = np.array(leftShoulderY)

    from matplotlib import pyplot as plt

    x1, y1 = leftElbowY.T
    plt.plot(x1, y1, color='green', linewidth=1, label="leftElbowY")
    plt.xlabel("Frame")
    plt.ylabel("Coordinates")
    plt.title("leftElbowY")
    plt.scatter(x1, y1, color='green', s=1)
    plt.show()

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    plt.xlabel("Frame")
    plt.ylabel("Coordinates")
    plt.title("leftElbowY vs rightElbowY")

    x1, y1 = rightElbowY.T
    plt.plot(x1, y1, color='blue', linewidth=1, label="rightElbowY")
    plt.scatter(x1, y1, color='blue', s=1)

    x2, y2 = leftElbowY.T
    plt.plot(x2, y2, color='green', linewidth=1, label="leftElbowY")
    plt.scatter(x1, y1, color='green', s=1)

    distance, path = fastdtw(leftElbowY, rightElbowY, dist=euclidean)
    for i in path:
        xcoord, ycoord = zip(leftElbowY[i[0]], rightElbowY[i[1]])
        plt.plot(xcoord, ycoord, color='orange', linewidth=0.6)
    plt.show()
    print(distance)
    print(path)
