
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os
from PIL import Image
import grad_cam_lib
IMAGE_SIZE = [300, 300]
MODEL = load_model('model.h5')




def ml_job(input_file):
    model_input_filepath = input_file
    img_read = load_img(model_input_filepath, target_size=(IMAGE_SIZE))
    img_array = img_to_array(img_read)
    im_PIL = Image.fromarray(img_array.astype(np.uint8))
    im_PIL.save('static/processed_image.jpg')

    img_array_norm = img_array / 255.0
    img_array_norm_batch = np.expand_dims(img_array_norm, axis=0)
    print(type(img_array_norm_batch))
    print(img_array_norm_batch.shape)
    pred_res_arr = MODEL.predict(img_array_norm_batch)
    assert pred_res_arr.ndim == 2
    assert pred_res_arr.shape[0] == 1
    prob_arr = pred_res_arr[0]
    pred_class = np.argmax(prob_arr)

    last_layer_index = -4
    last_conv_layer_name = MODEL.layers[last_layer_index].name
    classifier_layer_names = [x.name for x in MODEL.layers[last_layer_index+1:]]
    superimposed_img_arr = grad_cam_lib.perform_grad_cam(img_read=img_read, model=MODEL, 
                                                    last_conv_layer_name=last_conv_layer_name, 
                                                    classifier_layer_names=classifier_layer_names, 
                                                    image_size=IMAGE_SIZE)
    superimposed_img_arr_PIL = Image.fromarray(superimposed_img_arr.astype(np.uint8))
    superimposed_img_arr_PIL.save('static/gradcam.jpg')
    if pred_class == 0:
        pred_class_name = 'normal'
    elif pred_class == 1:
        pred_class_name = 'sick'
    else:
        raise ValueError('unknown predicted class')
    normal_prob = f'{prob_arr[0]*100:.2f} %'
    sick_prob = f'{prob_arr[1]*100:.2f} %'

    result = {}
    result['pred_class_name'] = pred_class_name
    result['normal_prob'] = normal_prob
    result['sick_prob'] = sick_prob
    # result['pred_class_name'] = pred_class_name
    # result['pred_class_name'] = pred_class_name
    # result['pred_class_name'] = pred_class_name
    return result


if __name__ == "main":
    result = ml_job('static/model_input.jpg')
    print(result)
