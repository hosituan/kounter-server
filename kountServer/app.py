import os
import numpy as np
import ntpath
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
#from .egg_kounter import startCountEggs
from eggKounter import startCountEggs
from woodKounter import startCountWood
from globalModel import GlobalModel
import cloudinary
import cloudinary.uploader
import cloudinary.api
import pickle
import os, shutil
from flask_socketio import SocketIO, emit
import downloadModel


import tensorflow as tf
import keras
from object_detector_retinanet.keras_retinanet import models
from tensorflow.python.keras.backend import get_session

UPLOAD_FOLDER = 'uploads/'
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads/')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


cloudinary.config(
  cloud_name = 'dggbuxa59',  
  api_key = '651855936159331',  
  api_secret = 'YZcmgha2qUntVE_QrxeCThMLJEM'  
)



#clean after upload
def clean():
  for file_name in os.listdir(OUTPUT_FOLDER):
    file_path = os.path.join(OUTPUT_FOLDER, file_name)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
  for file_name in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))



def upload_diary(filename):
  os.chdir(OUTPUT_FOLDER)
  respone = cloudinary.uploader.upload(filename, folder = "kount_result")
  os.chdir("..")
  clean()
  return respone['secure_url']

@app.route('/', methods=['GET', 'POST'])
def welcome():
  return "Hello From Ho Si Tuan - My Kounter"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("checking...")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify(
              success=False,
              message="No file",
            )
        file = request.files['file']
        print("got file")
        if file.filename == '':
              return jsonify(
              success=False,
              message="File name is blank",
            )
        if file and allowed_file(file.filename):
            print("allowed file")
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return jsonify(
            success=True,
            message="Upload image",
            fileName=file.filename,
            )
        return jsonify(
          success=False,
          message="Only accept PNG, JPG, JPEG extension"
        )
    return "This is GET method"

@app.route('/count', methods=['GET', 'POST'])
def count():
  if request.method == 'POST':
    if request.form.get("name") == "Chicken Egg":
      filename = str(request.form.get("imageName"))
      print("Start counting")
      showConfidence = False
      if int(request.form.get("showConfidence")) == 0:
        showConfidence = True
      startCountEggs(os.path.join(UPLOAD_FOLDER, filename), filename, showConfidence)
      result_file = str(filename + "_result.jpg")            
      read_dictionary = np.load(os.path.join(OUTPUT_FOLDER, filename+'_result.npy'),allow_pickle='TRUE').item()
      count_value = read_dictionary[filename]
      url = upload_diary(result_file)
      return jsonify(
        success=True,
        message="Counted",
        name = "Chicken Egg",
        fileName=filename,
        url=url,
        count=count_value
      )
    elif request.form.get("name") == "Fire Wood":
      filename = str(request.form.get("imageName"))
      print("Start counting")
      showConfidence = False
      if int(request.form.get("showConfidence")) == 0:
        showConfidence = True
      startCountWood(os.path.join(UPLOAD_FOLDER, filename), filename, showConfidence)
      result_file = str(filename + "_result.jpg")            
      read_dictionary = np.load(os.path.join(OUTPUT_FOLDER, filename+'_result.npy'),allow_pickle='TRUE').item()
      count_value = read_dictionary[filename]
      url = upload_diary(result_file)
      return jsonify(
        success=True,
        message="Counted",
        name = "Fire Wood",
        fileName=filename,
        url=url,
        count=count_value
      )
    return jsonify(
                success=False,
                message="We can't count this type!"
              )
  return jsonify(
                success=False,
                message="This is GET method"
              )


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

downloadModel.main()
tf.disable_resource_variables()
get_session()
# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
egg_model_path = os.path.join('object_detector_retinanet','weights', 'eggCounter_model.h5')
GlobalModel.eggModel = models.load_model(egg_model_path, backbone_name='resnet50')

wood_model_path = os.path.join('object_detector_retinanet','weights', 'woodCounter_model.h5')
GlobalModel.woodModel = models.load_model(wood_model_path, backbone_name='resnet50')

# model.summary()
print("loaded all model")
# socketio = SocketIO(app)

# if __name__ == "__main__":
#   socketio.run(app, host='0.0.0.0', port=80, debug=False,use_reloader=False)