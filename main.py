from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import base64

# Load the custom model
model_finetuned = load_model('model-finetuned.keras')

# Flask web app
app = Flask(__name__)


@app.route('/upload')
def display_upload_form():
    return """
    <html>
       <body>
          <form action = "/uploader" method = "POST"
             enctype = "multipart/form-data">
             <input type = "file" name = "file" />
             <input type = "submit"/>
          </form>
       </body>
    </html>
    """


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    # Load the weights and the filenames inside the function
    feature_list = pickle.load(open('features-caltech101-resnet-finetuned.pickle', 'rb'))
    filenames = pickle.load(open('filenames-caltech101.pickle', 'rb'))

    # Define the indexer with the correct metric
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(feature_list)

    if request.method == 'POST':
        img = Image.open(request.files['file'])
        img = img.resize((224, 224))  # resize the image to be 224 x 224
        img = np.array(img) / 255  # normalize the image
        img = np.expand_dims(img, axis=0)  # add a dimension to the np array : (1, 224, 224, 3)

        # Get feature list
        finetunned_cnn = Sequential()
        finetunned_cnn.add(model_finetuned.get_layer('resnet50'))
        finetunned_cnn.add(GlobalAveragePooling2D())
        feature_list = finetunned_cnn.predict(img)

        # Reshape the feature_list to ensure it's 2D
        feature_list = feature_list.reshape(1, -1)

        # Get the most similar feature list
        distances, indices = neighbors.kneighbors(feature_list)

        return render_template_string('''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Index</title>
            </head>
            <body>
                <img src="data:image/png;base64,{{ img1_str }}"/>
                <img src="data:image/png;base64,{{ img2_str }}"/>
                <img src="data:image/png;base64,{{ img3_str }}"/>
                <img src="data:image/png;base64,{{ img4_str }}"/>
                <img src="data:image/png;base64,{{ img5_str }}"/>
            </body>
        </html>
        ''', img1_str=base64.b64encode(open(filenames[indices[0][0]], 'rb').read()).decode("utf-8"),
                                      img2_str=base64.b64encode(open(filenames[indices[0][1]], 'rb').read()).decode(
                                          "utf-8"),
                                      img3_str=base64.b64encode(open(filenames[indices[0][2]], 'rb').read()).decode(
                                          "utf-8"),
                                      img4_str=base64.b64encode(open(filenames[indices[0][3]], 'rb').read()).decode(
                                          "utf-8"),
                                      img5_str=base64.b64encode(open(filenames[indices[0][4]], 'rb').read()).decode(
                                          "utf-8"))


@app.route('/')
def hello_world():
    return 'Hello Flask !!!'


@app.route('/greet')
def greet():
    return '<h1 style="color: red">Hi Antanas !!!</h1>'


if __name__ == '__main__':
    app.run()
