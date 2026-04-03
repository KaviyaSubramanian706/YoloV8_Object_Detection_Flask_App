from flask import Flask, render_template, request,send_from_directory
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        results = model(image)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

        output_path = os.path.join(OUTPUT_FOLDER, file.filename)
        cv2.imwrite(output_path, image)

        output_image = file.filename

    return render_template("index.html", output_image=output_image)

@app.route('/outputs/<filename>')
def display_output(filename):
    return send_from_directory('outputs', filename)

if __name__ == "__main__":
    app.run(debug=True)