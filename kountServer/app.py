import os
import numpy as np
import ntpath
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
#from .egg_kounter import startCountEggs
from eggKounter import startCountEggs
from woodKounter import startCountWood
from steelKounter import startCountSteel
from kountObject import startCount
from globalModel import GlobalModel
from CountObject import CountObject
import cloudinary
import cloudinary.uploader
import cloudinary.api
import pickle
import os, shutil
from flask_socketio import SocketIO, emit
import downloadModel
import _thread

import tensorflow as tf
import keras
from object_detector_retinanet.keras_retinanet import models
from tensorflow.python.keras.backend import get_session
import json

UPLOAD_FOLDER = 'uploads/'
app = Flask(__name__)
socketio = SocketIO(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads/')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/', methods=['GET', 'POST'])
def welcome():
  return "Hello From Ho Si Tuan - My Kounter"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/add', methods =['GET', 'POST'])
def add_object():
  print("Adding")
  if request.method == 'POST':
    id = request.form.get('id')
    name = request.form.get('name')
    driveID = request.form.get('driveID')    
    data = {}
    data['countObject'] = []
    if os.path.isfile('countObjects.txt'):
        print ("Object list exist")
        with open('countObjects.txt') as json_file:
          data = json.load(json_file)
          for o in data['countObject']:
            print(o)
    else:
        print ("Object list not exist")
    data['countObject'].append({
        'id': id,
        'name': name,
        'driveID': driveID
      })
    with open('countObjects.txt', 'w') as outfile:
      json.dump(data, outfile)
      return jsonify(
          success=True,
          message="Add to object list"
        )
    return jsonify(
    success=False,
    message="Something went wrong"
  )

  else:
    return jsonify(
              success=False,
              message="This is GET method"
            )

@app.route('/prepare', methods = ['GET', 'POST'])
def prepare():
  if request.method == 'POST':
    print('preparing')
    downloadModel.main(objectList)
    print('downloaded all model')
    loadObjects()
    objectID = request.form.get('id')
    for obj in objectList:
      print(obj.id)
      if obj.id == objectID:
        objName = obj.name
        modelName = objName + '_model.h5'
        modelPath = os.path.join('object_detector_retinanet','weights', modelName)
        # start tensorflow backend
        tf.disable_resource_variables()
        get_session()
        # set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(get_session())
        GlobalModel.model = models.load_model(modelPath, backbone_name='resnet50')
        GlobalModel.graph = tf.get_default_graph()
        print("Loaded model")
        return jsonify(
              success=True,
              message="Prepared model"
            )
  return jsonify(
              success=False,
              message="This is GET method"
            )

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
    return jsonify(
                success=False,
                message="This is GET method"
              )

@app.route('/count', methods=['GET', 'POST'])
def count():
  print("checking...")
  if request.method == 'POST':
    if 'file' not in request.files:
        return jsonify(
          success=False,
          message="No file",
        )
    file = request.files['file']
    print("got file")
    socketio.emit('countResult', {
      'success': True,
      'message': 'Got file'
    })
    
    if file.filename == '':
          return jsonify(
          success=False,
          message="File name is blank",
        )
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(UPLOAD_FOLDER, filename))
      idObject = request.form.get("id")
      typeObject = request.form.get("name")
      print("Start counting " + typeObject)
      socketio.emit('countResult', {
        'success': True,
        'message': 'Start counting'
      })
      result = startCount(os.path.join(UPLOAD_FOLDER, filename))           
      return jsonify(
        success=True,
        message="Counted",
        name = typeObject,
        fileName=filename,
        result = result
      )
      clean()
    return jsonify(
      success=False,
      message="Only accept PNG, JPG, JPEG extension"
    )
  return jsonify(
                success=False,
                message="This is GET method"
              )

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def loadObjects():
    if os.path.isfile('countObjects.txt'):
        print ("Object list exist")
        with open('countObjects.txt') as json_file:
          data = json.load(json_file)
          for countObj in data['countObject']:
            obj = CountObject(countObj['id'], countObj['name'], countObj['driveID'])
            objectList.append(obj)
          print ("Loaded")
    else:
        print ("Object list not exist")



# get object list
objectList = []
loadObjects()
# download model
downloadModel.main(objectList)
# egg_model_path = os.path.join('object_detector_retinanet','weights', 'eggCounter_model.h5')
# GlobalModel.eggModel = models.load_model(egg_model_path, backbone_name='resnet50')
# wood_model_path = os.path.join('object_detector_retinanet','weights', 'woodCounter_model.h5')
# GlobalModel.woodModel = models.load_model(wood_model_path, backbone_name='resnet50')
# steel_model_path = os.path.join('object_detector_retinanet','weights', 'steelCounter_model.h5')
# GlobalModel.steelModel = models.load_model(steel_model_path, backbone_name='resnet50')


if __name__ == "__main__":
  socketio.run(app, host='0.0.0.0', port=80, debug=False,use_reloader=False)