import cv2
import mediapipe as mp

# Face Meshの初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 描画用
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# カメラ起動
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    # 左右反転（鏡のような表示）
    img = cv2.flip(img, 1)

    # BGR → RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face Mesh実行
    results = face_mesh.process(rgb_img)

    # 顔が見つかった場合
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Face Mesh描画
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # 表示
    cv2.imshow("Face Mesh", img)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()