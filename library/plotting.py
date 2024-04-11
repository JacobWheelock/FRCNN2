import cv2
import matplotlib.pyplot as plt
from .utils import getDataset

# function to visualize a single sample
def visualize_sample(image, target, classes):
    for i in range(0,len(target['boxes'])):
        box = target['boxes'][i]
    
        label = classes[target['labels'][i]]
    
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
            )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        plt.imshow((image*255).astype('uint8'))

def visualize_samples(NUM_SAMPLES, TRAIN_DIR, RESIZE_TO, CLASSES):
    dataset = getDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    for i in range(NUM_SAMPLES):
        image, target = dataset[i]
        plt.figure()
        visualize_sample(image, target, CLASSES)