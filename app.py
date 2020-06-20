import os
import sys
import shutil
import glob

# 플라스크 모듈
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import darkflow.net.yolov2.predict

import test

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('index.html')


detect_temp = 0


# 파일 업로드 처리
@app.route('/upload', methods = ['POST'])
def upload():
    if os.path.isdir('static'):
        files = glob.glob('static/*')
        if os.path.isdir('static/out'):
            shutil.rmtree('static/out')
        for f in files:
            if os.path.splitext(f)[1] in ['.jpeg', '.jpg', '.png', 'bmp', 'gif']:
                os.remove(f)
    if request.method == 'POST':
        target = os.path.join(APP_ROOT, 'static')
        print(target)   # /Users/NoHack/Desktop/capstone/images/

        if not os.path.isdir(target):
            os.mkdir(target)
        else:
            print("Couldn't create upload directory: {}".format(target))
        print('파일들: ' + str(request.files.getlist("file")))
        filenames = []
        for upload in request.files.getlist("file"):
            print('업로드 정보: ' + str(upload))
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join([target, filename])
            print('Accept incoming file: ', filename)
            print('Save it to: ', destination)
            upload.save(destination)

            print('파일 이름: ' + filename)
            filenames.append(filename)

        dst = 'ckpt'

        # 홍채
        darkflow.net.yolov2.predict.detection_choice = 2
        src = 'ckpt/ck_pupil/checkpoint'
        shutil.copy(src, dst)
        test.detect()

        # 지문
        darkflow.net.yolov2.predict.detection_choice = 1
        src = 'ckpt/ck_finger/checkpoint'
        shutil.copy(src, dst)
        test.detect()

        darkflow.net.yolov2.predict.detection_choice = 0
        # return render_template("complete.html", image_name=filename)

        return render_template("complete.html", image_name=filenames)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory('image', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')