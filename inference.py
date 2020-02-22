''' Runs inference on a given GeoTIFF image.

example:
$ python inference.py --checkpoint_path checkpoints/cp.135.ckpt \
    --image_path sample_data/sentinel2_example.tif --save_path water_map.png
'''

# Uncomment this to run inference on CPU if your GPU runs out of memory
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
import argparse
import deepwatermap
import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
from download_data import download_from_gdrive
from projection import set_projection


# check the existence of pre-trained model and test data. 
def check_data_exists(work_dir):
    checkpoints = os.path.join(work_dir,"checkpoints.zip")
    metadata = os.path.join(work_dir, "metadata.zip")
    sample_data = os.path.join(work_dir, "sample_data.zip")

    checkpoints_url = "https://drive.google.com/file/d/1WdFa0O10Wt955tmvGbGykjIcJvRZBHVL"
    metadata_url = "https://drive.google.com/file/d/1AIiYBdFFG3fwaYnpw8oDC6lHQlybuJ0y"
    sample_data_url = "https://drive.google.com/file/d/1t9bUeg53wqEtsptCW1GbQnds0s5mCgW9"

    if not os.path.exists(checkpoints):
        file_name = os.path.basename(checkpoints)
        print("The pre-trained model does not exist. Downloading {} (131 MB) ...".format(file_name))
        download_from_gdrive(checkpoints_url, file_name, unzip=True)

    if not os.path.exists(metadata):
        file_name = os.path.basename(metadata)
        print("The metadata does not exist. Downloading {} (45 MB) ...".format(file_name))
        download_from_gdrive(metadata_url, file_name, unzip=True)

    if not os.path.exists(sample_data):
        file_name = os.path.basename(sample_data)
        print("The test data do not exist. Downloading {} (472 MB) ...".format(file_name))
        download_from_gdrive(sample_data_url, file_name, unzip=True)


# padding the upper-left
def find_padding(v, divisor=32):
    v_divisible = max(divisor, int(divisor * np.ceil( v / divisor )))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2

def predict(checkpoint_path, image_path, save_path):
    # load the model
    model = deepwatermap.model()
    model.load_weights(checkpoint_path)

    # load and preprocess the input image
    image = tiff.imread(image_path)
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')
    # plt.imshow(image[:, :, 0])
    # plt.show()
    image = image.astype(np.float32)
    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    dwm = model.predict(image)
    dwm = np.squeeze(dwm)   #Remove single-dimensional entries from the shape of an array
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]  #remove padding from prediction image

    # soft threshold
    dwm = 1./(1+np.exp(-(16*(dwm-0.5))))
    dwm = np.clip(dwm, 0, 1)  # Given an interval, values outside the interval are clipped to the interval edges

    # save the output water map
    cv2.imwrite(save_path, dwm * 255)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_path', type=str,
    #                     help="Path to the dir where the checkpoints are stored")
    # parser.add_argument('--image_path', type=str, help="Path to the input GeoTIFF image")
    # parser.add_argument('--save_path', type=str, help="Path where the output map will be saved")
    # args = parser.parse_args()
    # main(args.checkpoint_path, args.image_path, args.save_path)

    # python inference.py --checkpoint_path checkpoints/cp.135.ckpt --image_path sample_data/sentinel2_example.tif --save_path results/water_map.png
    # python inference.py --checkpoint_path checkpoints/cp.135.ckpt --image_path sample_data/p225r060.tif --save_path results/water_map2.png

    # Customized input image and output path
    ################################################################
    work_dir = os.path.dirname(os.path.abspath(__file__))
    
    # download pre-trained model and sample data
    check_data_exists(work_dir)

    # set input image name/path
    in_image_name = "sentinel2_example.tif"
    # in_image_name = "p225r060_WC_19991031.tif"
    in_image_name = "p200r018_WC_19990712.tif"
    in_image_path = os.path.join(work_dir, "sample_data", in_image_name)
    
    # whether or not to use PNG as the output image format
    use_png_ext = False

    # set output image folder
    output_dir = os.path.join(os.path.expanduser("~"), "temp")
    # output_dir = os.path.join(work_dir, "results")

    ################################################################

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output image name is based on the input image name
    basename, extension = os.path.splitext(in_image_name)
    if use_png_ext:
        extension = ".png"
    out_image_name = basename + "_water" + extension
    out_image_path = os.path.join(output_dir, out_image_name)

    # load pre-trained model
    checkpoint_path = os.path.join(work_dir, "checkpoints/cp.135.ckpt")

    # image classification using deep learning
    predict(checkpoint_path, in_image_path, out_image_path)

    # add projection coordinate system
    if not use_png_ext:
        set_projection(out_image_path, template_raster=in_image_path)

    print("\nOutput path: {}".format(out_image_path))