import cv2
import matplotlib.pyplot as plt
from .utils import getDataset

def visualize_sample(TRAIN_DIR, RESIZE_TO, CLASSES, index):
    dataset = getDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    
    image, target = dataset[index]
    fig = plt.figure()
    ax = fig.subplots()
    
    for i in range(0,len(target['boxes'])):
        box = target['boxes'][i]
        classesBG = ['background'] + CLASSES
        label = classesBG[target['labels'][i].item()]
    
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
            )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
    plt.imshow(image)
    return fig