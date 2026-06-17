import cv2
import mediapipe as mp

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 顔検出の設定
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 顔検出
    results = face_detection.process(img_rgb)

    # 顔が見つかった場合
    if results.detections:
        for detection in results.detections:

            # 顔に枠を描画
            mp_drawing.draw_detection(img, detection)

    # 画面表示
    cv2.imshow("Face Detection", img)

    # ESCキーで終了
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()