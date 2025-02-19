from Utils import *
from YOLOv8SR import YOLOv8SR
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import pandas as pd 
import matplotlib.pyplot as plt
import os, random, time
import matplotlib.pyplot as plt
import tensorflow as tf
from TensorSR import *


# Parameters  python3 -m pip install tensorflow-macos
Epochs = 300    # same as YOLOv5 paper
cfg ='train.yaml'
weights = "yolov8n.pt"
resultsPath = 'runs/train'
ds_path = 'datasets\\VisDrone\\val'
ds_img_path = f'{ds_path}\\images'
image_size=640
optimizer='AdamW'
verbose = False
AdamW_weight_decay = 0.0005
lr0 = 0.01	
pretrained=True

image = "low-res-car.jpg"
super_res_model = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

# set YOLO logging
def set_logging(name=None, verbose=False):
    # rank for Multi-GPU trainings
    rank = int(os.getenv('RANK', -1))  
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

# Clean up existing runs
#Pre_Process(delete_runs = False)

#  set logger
LOGGER = set_logging(__name__) 

# Load a model
yolov8sr = YOLOv8SR(cfg = cfg, pretrained_weights=weights )
yolov8sr = yolov8sr.yolo


# Super Resolution
highres_img = image2tensor(image)

lowres_img = downscale_tensor(tf.squeeze(highres_img))

# Plotting Low Resolution Image
#show_tensor_img(tf.squeeze(lowres_img), title="Low Resolution")

sr_model = load_sr_model(super_res_model)

start_time = time.time()
SR_image = sr_model(lowres_img)
SR_image = tf.squeeze(SR_image)
print("Time Taken: %f" % (time.time() - start_time))

#show_tensor_img(tf.squeeze(SR_image), title="Super Resolution")

psnr = tf.image.psnr(
    tf.clip_by_value(SR_image, 0, 255),
    tf.clip_by_value(highres_img, 0, 255), max_val=255)

ssim = tf.image.ssim(
    tf.clip_by_value(SR_image, 0, 255),
    tf.clip_by_value(highres_img, 0, 255), max_val=255)


print(f"PSNR Achieved: {psnr.numpy()[0]:0.3f}")
print(f"SSIM Achieved: {ssim.numpy()[0]:0.3f}")


fig, axes = plt.subplots(1, 2)
plt.subplot(121)
show_tensor_img(tf.squeeze(lowres_img), "Downsampled Input Image to model")
plt.subplot(122)
show_tensor_img(tf.squeeze(SR_image), "Super Resolution PSNR : %.2f" % psnr)
plt.savefig("Results.jpg", bbox_inches="tight")
print("PSNR: %f" % psnr)
plt.show(block=True)



def tanslate_bbox(size, bbox):
    # Convert VisDrone box to YOLO xywh box
    d_w = 1. / size[0]
    d_h = 1. / size[1]
    return (bbox[0] + bbox[2] / 2) * d_w, (bbox[1] + bbox[3] / 2) * d_h, bbox[2] * d_w, bbox[3] * d_h


def convert_annotations(dir):
    dir = Path(dir)
    (dir / 'labels').mkdir(parents=True, exist_ok=True)  
    annotations = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for anotation in annotations:
        img_size = Image.open((dir / 'images' / anotation.name).with_suffix('.jpg')).size
        ann_lines = []
        with open(anotation, 'r') as file:  
            for ann_row in [x.split(',') for x in file.read().strip().splitlines()]:
                if ann_row[4] == '0':  
                    continue
                cls = int(ann_row[5]) - 1
                box = tanslate_bbox(img_size, tuple(map(int, ann_row[:4])))
                ann_lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(anotation).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as lableFile:
                    lableFile.writelines(ann_lines)  # write label.txt


# convert VisDrone annotations to YOLO labels
for annotation in [ds_path]:
    convert_annotations(annotation)  

def on_train_epoch_end(trainer):
  #print("epoch ... ")
  clear = lambda: os.system('cls')
  clear()


# # Train the YOLOv8SR model
results = yolov8sr.train(data='train.yaml', epochs=Epochs, imgsz=image_size, \
                         optimizer=optimizer,verbose = verbose,weight_decay = AdamW_weight_decay, \
                            lr0 =lr0,pretrained = True )

# Post Process results from the runs 
Post_Process(delete_runs = False)


## Training Results 
results_df = pd.read_csv('results.csv')

# read results csv file 
names = results_df.columns
names = [name.replace('/','').strip() for name in names]
results_df.columns = names
results_df.columns

plt.figure()
plt.plot(results_df['epoch'],results_df['trainbox_loss'],marker="o")
plt.title('Box Loss')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('box loss')
plt.show(block=False)


plt.figure()
plt.plot(results_df['epoch'],results_df['traindfl_loss'],marker="o",color="red")
plt.title('Yolov8-SR DFL (distributional focal loss)')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('dfl loss')
plt.show(block=False)


plt.figure()
plt.plot(results_df['epoch'],results_df['lrpg0'],marker="o",color="red")
plt.title('learning rate of the Yolov8-SR')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('lr')
plt.show(block=False)


plt.figure()
plt.plot(results_df['epoch'],results_df['metricsprecision(B)'],marker="+",color="k")
max_Precision = results_df['metricsprecision(B)'].max()
plt.title(f'Precision of the Yolov8-SR max Precision = {max_Precision}')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('precision')
plt.show(block=False)



# mAP 
results_df['metricsmAP50(B)'].max()
mAP,_ = get_mAP(results_df)
mAP_95,_ = get_mAP(results_df,mAP95=True)


plt.figure()
plt.plot(results_df['epoch'],mAP,marker="o",color="b")
plt.plot(results_df['epoch'],mAP_95,marker="o",color="r")
plt.title(f'mAP50(%)  = {mAP.max():.2f}, mAP50-95(%) = {mAP_95.max():.2f}')
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('lr')
plt.show(block=False)

plt.figure()
labels = plt.imread('labels.jpg')
plt.imshow(labels)
plt.title("Classes of the VisDrone Dataset")
plt.show(block=False)


plt.figure()
plt.subplot(1, 2, 1)
labels = plt.imread('val_batch0_labels.jpg')
plt.imshow(labels)
plt.title('Validation lables')
plt.subplot(1, 2, 2)
pred = plt.imread('val_batch0_pred.jpg')
plt.imshow(pred)
plt.title('Validation predictions')
plt.show(block=False)


plt.figure()
labels = plt.imread('train_batch0.jpg')
plt.imshow(labels)
plt.title("training batch labels")
plt.show(block=False)

plt.figure()
labels = plt.imread('labels_correlogram.jpg')
plt.imshow(labels)
plt.title("labels correlogram")
plt.show(block=False)



## predicting 
yolov8sr = YOLOv8SR(weights)

# randomly select file from the validation/test dataset
print('Randomly selecting file from the database for prediction')
File = ds_img_path + "//" + random.choice(os.listdir(ds_img_path)) 


# visualize, show_boxes
yolo_results = yolov8sr.predict(File,save=True,show_boxes=True,show=True)

yolov8_result = yolo_results[0]

len(yolov8_result.boxes)
bbox = yolov8_result.boxes[0]

for bbox in yolov8_result.boxes:
  class_id = yolov8_result.names[bbox.cls[0].item()]
  cords = bbox.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(bbox.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")


plt.figure()
labels = plt.imread(File)
plt.imshow(labels)
plt.title("VisDrone Sample for Prediction")
plt.show(block=True)
