import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from data import process_image_file
from utils import get_unique_filename, allowed_file_extensions, upload_to_s3

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

app = Flask(__name__)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


# instead of Flask_app use /content/gdrive/My Drive/Allps/COVID-Net/working-pre-trained-weights
# os.chdir('/root/Flask_app')

@app.route('/')
def index():
    return render_template('index.html')


def perform_inference(file_path):
    # a=request.form['featurea'] image name input
    # Inference.py
    # Inference.py

    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join('models/COVIDNet-CXR3-B', 'model.meta'))
    saver.restore(sess, os.path.join('models/COVIDNet-CXR3-B', 'model-1014'))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name('input_1:0')
    pred_tensor = graph.get_tensor_by_name('norm_dense_1/Softmax:0')

    x = process_image_file(file_path, 0.08, 480)
    x = x.astype('float32') / 255.0
    pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
    all_predictions = ('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))
    # send the values like render_template('gout.html',values)
    predicted_output = inv_mapping[pred.argmax(axis=1)[0]]
    return predicted_output, all_predictions


@app.route('/upload-image', methods=['POST'])
def handleUpload():
    if 'x_ray_image' not in request.files:
        print('no file found in upload request')
        return jsonify("{'message': 'no file in upload request'}")
    file = request.files['x_ray_image']

    if file.filename == '':
        return jsonify({'message': 'No file name provided'})

    file.filename = get_unique_filename() + '.' + file.filename.split('.')[-1]

    if file and allowed_file_extensions(file.filename):
        if not os.path.exists(os.getenv('UPLOAD_FOLDER')):
            os.makedirs(os.getenv('UPLOAD_FOLDER'))
        uploaded_file_path = os.path.join(os.getenv('UPLOAD_FOLDER'), file.filename)
        file.save(uploaded_file_path)
        prediction, all_predictions = perform_inference(uploaded_file_path)
        s3_url = upload_to_s3(uploaded_file_path, str(prediction) + '__' + str(file.filename))

        print('*****************')
        print('*****************')
        print('*****************')
        print(prediction)
        print('*****************')
        print('*****************')
        print('*****************')

        return app.response_class(
            response=json.dumps({"prediction": str(prediction), "file_url": str(s3_url), "all": all_predictions}),
            status=200,
            mimetype='application/json'
        )
    else:
        return app.response_class(
            response=json.dumps({"message": "invalid file provided"}),
            status=500,
            mimetype='application/json'
        )


if __name__ == "__main__":
    app.run(debug=True)
