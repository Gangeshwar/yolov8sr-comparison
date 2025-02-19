import matplotlib.pyplot as plt
import os
import tensorflow_hub as hub
import numpy as np
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
import tensorflow as tf
from PIL import Image


def image2tensor(img):
  input_img = tf.image.decode_image(tf.io.read_file(img))
  if input_img.shape[-1] == 4:
    input_img = input_img[...,:-1]
  hr_size = (tf.convert_to_tensor(input_img.shape[:-1]) // 4) * 4
  input_img = tf.image.crop_to_bounding_box(input_img, 0, 0, hr_size[0], hr_size[1])
  input_img = tf.cast(input_img, tf.float32)
  return tf.expand_dims(input_img, 0)

def write_tensor_image(tf_img, FilePath):
  if not isinstance(tf_img, Image.Image):
    tf_img = tf.clip_by_value(tf_img, 0, 255)
    tf_img = Image.fromarray(tf.cast(tf_img, tf.uint8).numpy())
  tf_img.save("%s.jpg" % FilePath)
  print("Saved as %s.jpg" % FilePath)


def show_tensor_img(tf_img, title=""):
  tf_img = np.asarray(tf_img)
  tf_img = tf.clip_by_value(tf_img, 0, 255)
  tf_img = Image.fromarray(tf.cast(tf_img, tf.uint8).numpy())  
  plt.imshow(tf_img)
  plt.axis("off")
  plt.title(title)
  
  
# Defining helper functions
def downscale_tensor(tensor_img,factor = 4):
  img_size = []
  if len(tensor_img.shape) == 3:
    img_size = [tensor_img.shape[1], tensor_img.shape[0]]
  else:
    raise ValueError("Dimension mismatch. Can work only on single image.")

  tensor_img = tf.squeeze(
      tf.cast(
          tf.clip_by_value(tensor_img, 0, 255), tf.uint8))

  lowres_img = np.asarray(
    Image.fromarray(tensor_img.numpy())
    .resize([img_size[0] // factor, img_size[1] // factor],
              Image.BICUBIC))

  lowres_img = tf.expand_dims(lowres_img, 0)
  lowres_img = tf.cast(lowres_img, tf.float32)
  return lowres_img


def load_sr_model(model = ""):
  SR_Model = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
  sr_model = hub.load(SR_Model)
  return sr_model
  