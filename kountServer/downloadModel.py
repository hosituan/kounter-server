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

def main():
    egg_file_id = '129sQfXjY4doJmMcY1a-ZBRGEzJneuqNR'
    egg_model_path = os.path.join('object_detector_retinanet','weights', 'eggCounter_model.h5')
    if os.path.isfile(egg_model_path):
        print ("egg Model exist")
    else:
        print("Downloading egg model...")
        download_file_from_google_drive(egg_file_id, egg_model_path)

    wood_file_id = '11NZf3kDjSX6xPkWLMRaLnG2U1yKSDRv7'
    wood_model_path = os.path.join('object_detector_retinanet','weights', 'woodCounter_model.h5')
    if os.path.isfile(wood_model_path):
        print ("wood Model exist")
    else:
        print("Downloading wood model...")
        download_file_from_google_drive(wood_file_id, wood_model_path)
    
    wood_file_id = '1-kGFubcIFVfgUkhYlrOzRaUW5kq0Fm7x'
    wood_model_path = os.path.join('object_detector_retinanet','weights', 'steelCounter_model.h5')
    if os.path.isfile(wood_model_path):
        print ("steel Model exist")
    else:
        print("Downloading steel model...")
        download_file_from_google_drive(wood_file_id, wood_model_path)
    
