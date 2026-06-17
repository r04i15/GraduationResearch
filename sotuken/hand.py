import cv2
import mediapipe as mp

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 描画用
mp_draw = mp.solutions.drawing_utils

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 手を検出
    results = hands.process(img_rgb)

    # 手が見つかったら描画
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking", img)

    # ESCキーで終了
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()