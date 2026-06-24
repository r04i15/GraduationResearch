import cv2
import mediapipe as mp
import math

# Face Mesh初期化
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# カメラ起動
cap = cv2.VideoCapture(0)

previous_direction = ""

while True:
    success, img = cap.read()

    if not success:
        break

    # 鏡表示
    img = cv2.flip(img, 1)

    # BGR → RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face Mesh実行
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = img.shape

            nose = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]

            x1 = int(left_cheek.x * w)
            y1 = int(left_cheek.y * h)

            x2 = int(right_cheek.x * w)
            y2 = int(right_cheek.y * h)

            face_width = math.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )
                        
            REFERENCE_DISTANCE = 50
            REFERENCE_FACE_WIDTH = 250

            distance_cm = (
                REFERENCE_FACE_WIDTH *
                REFERENCE_DISTANCE /
                face_width
            )

            nose_x = nose.x
            left_x = left_eye.x
            right_x = right_eye.x

            center = (left_x + right_x) / 2

            if nose_x < center - 0.02:
                direction = "LEFT"

            elif nose_x > center + 0.02:
                direction = "RIGHT"

            else:
                direction = "CENTER"

            cv2.putText(
                img,
                direction,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            if direction != previous_direction:
                print(direction)
                previous_direction = direction

            cv2.putText(
                img,
                f"Distance: {distance_cm:.1f} cm",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
        )

            # 全ランドマークに番号表示
            for idx, landmark in enumerate(face_landmarks.landmark):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.putText(
                    img,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1
                )

    cv2.imshow("Face Mesh Landmark Number", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()