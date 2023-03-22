import os
import mimetypes

from flask_cors import CORS
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import torchvision
import keras.utils as image
from PIL import Image
from functions import *

app_dir = os.getcwd()
UPLOAD_FOLDER = "WebApp/uploaded_file"
SEGMENTED_FOLDER = "WebApp/segmented_file"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER
CORS(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            
            img = Image.open(file_path)
            img_segmented = predict("segmentation", img)
            seg_path = os.path.join(app.config['SEGMENTED_FOLDER'],filename)
            torchvision.utils.save_image(img_segmented, seg_path)


            img = image.load_img(seg_path, target_size=(224, 224))
            classification = predict("classification", img)
            return classification
            

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("Current Workdir: " + app_dir)
    print("jsp: "+ app.config['UPLOAD_FOLDER'])
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/segmented/<filename>')
def segmented_file(filename):
    print("Current Workdir: " + app_dir)
    print("jsp: "+ app.config['SEGMENTED_FOLDER'])
    return send_from_directory(app.config['SEGMENTED_FOLDER'], filename)

@app.route('/classification', methods=['GET', 'POST'])
def classificator():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(224, 224))
            classe = classificatieur(img)

            return classe
        
        
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run()
