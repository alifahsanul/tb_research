# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging

from flask import Flask
from keras.models import load_model

app = Flask(__name__)

print('alif aik')

## ----------------- model -------------------------##
image_size = [300, 300]
MODEL = load_model('model.h5')
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
def my_predict(model, file):
    x = load_img(file, target_size=(image_size))
    x = img_to_array(x)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    assert array.ndim == 2
    assert array.shape[0] == 1
    prob_arr = array[0]
    answer = np.argmax(prob_arr)
    return answer, prob_arr
input_filepath = os.path.join(r'D:\tb_classification\data_prep\data_for_modelling\test\normal', 'CHNCXR_0008_0.png')
predicted_result = my_predict(MODEL, file=input_filepath)
print(predicted_result)
print('model loaded -----------------------------------------------------')




@app.route('/')
def hello():
    stringlist = []
    MODEL.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    """Return a friendly HTTP greeting."""
    return_str = "1122 Hello World!\naaa" + short_model_summary
    return return_str


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]