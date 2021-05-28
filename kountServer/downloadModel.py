import requests
import os
import os.path
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main(objectList):
    for obj in objectList:
        driveID = obj.driveID
        objName = obj.name
        modelName = obj.id + "_" + objName + '_model.h5'
        modelPath = os.path.join('object_detector_retinanet','weights', modelName)
        if os.path.isfile(modelPath):
            print(modelName + ' is exist')
        else:
            print("Downloading " + modelName) 
            download_file_from_google_drive(driveID, modelPath)

