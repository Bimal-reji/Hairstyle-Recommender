import os
import cv2
import uuid
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request, redirect
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_FOLDER = 'static/annotated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---- Dummy Training Data ----
X_train = [
    [0.8, 1.0, 1.2], [0.9, 1.0, 1.5],
    [1.0, 1.1, 1.1], [0.7, 1.0, 0.9],
    [0.6, 1.2, 1.4]
]
y_train = ['Round', 'Oval', 'Square', 'Heart', 'Diamond']
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# ---- Haircut Database ----
haircut_db = {
    'Round': ['Textured Crop', 'Messy Fringe', 'Angular Fringe', 'Undercut with Volume', 'Faux Hawk'],
    'Oval': ['Buzz Cut', 'Curtain Hair', 'Quiff', 'Taper Fade with Design', 'Bro Flow'],
    'Square': ['Pompadour', 'Side Part with Fade', 'French Crop', 'Ivy League Cut', 'Undercut with Sharp Edges'],
    'Heart': ['Messy Quiff', 'Low Fade with Fringe', 'Tapered Brush Up', 'Disconnected Undercut', 'Side Swept Bangs'],
    'Diamond': ['Layered Fringe', 'Brush Up with Mid Fade', 'Wavy Crop', 'Faux Hawk Fade', 'Slick Back']
}

def analyze_face(image, annotated_path=None):
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None, [], None

        landmarks = result.multi_face_landmarks[0].landmark
        jaw = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) - np.array([landmarks[454].x, landmarks[454].y]))
        cheek = np.linalg.norm(np.array([landmarks[93].x, landmarks[93].y]) - np.array([landmarks[323].x, landmarks[323].y]))
        length = np.linalg.norm(np.array([landmarks[10].x, landmarks[10].y]) - np.array([landmarks[152].x, landmarks[152].y]))

        jaw_ratio = jaw / cheek
        length_ratio = length / cheek
        features = [jaw_ratio, 1.0, length_ratio]
        shape = knn.predict([features])[0]
        styles = haircut_db.get(shape, ['Classic Cut'])

        # Draw green mesh on the face
        if annotated_path:
            annotated_img = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_img,
                landmark_list=result.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
            cv2.imwrite(annotated_path, annotated_img)

        return shape, styles, annotated_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect("/")
    file = request.files["image"]
    if file.filename == "":
        return redirect("/")

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{filename}")
    file.save(filepath)

    image = cv2.imread(filepath)
    shape, styles, _ = analyze_face(image, annotated_path=annotated_path)

    return render_template("index.html", shape=shape, styles=styles, image_path=annotated_path)

@app.route("/capture")
def capture():
    cap = cv2.VideoCapture(0)
    print("📷 Press SPACE to capture your face.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            break
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return redirect("/")
    cap.release()
    cv2.destroyAllWindows()

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    annotated_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{filename}")
    cv2.imwrite(filepath, frame)

    shape, styles, _ = analyze_face(frame, annotated_path=annotated_path)

    return render_template("index.html", shape=shape, styles=styles, image_path=annotated_path)

if __name__ == "__main__":
    app.run(debug=True)
