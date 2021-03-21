import os
import numpy as np
import ntpath
from os.path import join, dirname, realpath
from flask import Flask, flash, request, redirect, url_for, send_from_directory, send_file
from flask import jsonify
from werkzeug.utils import secure_filename
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads/')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


import cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api
import pickle

cloudinary.config(
  cloud_name = 'dggbuxa59',  
  api_key = '651855936159331',  
  api_secret = 'YZcmgha2qUntVE_QrxeCThMLJEM'  
)

def upload_diary(filename):
  OUTPUT_FOLDER = os.path.join(APP_ROOT, 'output/')
  os.chdir(OUTPUT_FOLDER)
  respone = cloudinary.uploader.upload(filename, folder = "kount_result")
  os.chdir("..")
  return respone['url']



@app.route('/', methods=['GET', 'POST'])
def welcome():
  return "Hello From Ho Si Tuan"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify(
              success=False,
              message="No file",
            )
        file = request.files['file']
        if file.filename == '':
              return jsonify(
              success=False,
              message="File name is blank",
            )

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #colorize
            return jsonify(test = "a"
            command = "python3 egg_kounter.py " + UPLOAD_FOLDER + "/" + filename
            print(command)
            print(os.path.abspath(os.getcwd()))
            os.system(command)
            result_file = str(filename + "_result.jpg")            
            url = upload_diary(result_file)
            path = 'output/'
            read_dictionary = np.load(os.path.join(path,filename+'_result.npy'),allow_pickle='TRUE').item()
            count_value = read_dictionary[filename]
            return jsonify(
              success=True,
              message="File name is uploaded",
              fileName=file.filename,
              path=UPLOAD_FOLDER,
              script=command,
              url=url,
              count=count_value
            )


    return "This is GET method"



# if __name__ == "__main__":
#   port = int(os.environ.get("PORT", 5000))
#   app.run(host='0.0.0.0', port=port, debug=True)


