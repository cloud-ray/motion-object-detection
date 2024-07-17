from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def gen_video_stream():
    url = "http://192.168.4.24:8080/video"
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/video")
def video_feed():
    return Response(gen_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=8079)
