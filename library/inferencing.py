import numpy as np
import torch
import cv2
import glob as glob
from .training import create_model

def load_model(model_name, MODEL_DIR, NUM_CLASSES):
    # set the computation device
    modelPath = './models/' + model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        modelPath, map_location=device
    ))
    model.eval()
    return model


def inference_video(DIR_TEST, OUT_DIR, vidName, model, detection_threshold, CLASSES):
    vid = cv2.VideoCapture(DIR_TEST)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    NUM_FRAMES = int(cv2.VideoCapture.get(vid, property_id))
    idx = 1
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter((OUT_DIR + '/' + vidName),cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    classes = [None] * NUM_FRAMES
    bboxes = [None] * NUM_FRAMES
    sscores = [None] * NUM_FRAMES
    
    while vid.isOpened():
        ret, image = vid.read()
        
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            sscores[idx] = scores
            
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            bboxes[idx] = boxes
            draw_boxes = bboxes[idx].copy() 
             
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_classes = np.array(pred_classes)
            pred_classes = pred_classes[scores >= detection_threshold]
            classes[idx] = pred_classes
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, str(pred_classes[j]), 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            out.write(orig_image)
        idx += 1
        print(f"Image {idx+1} done...")
        print('-'*50)
        if idx == NUM_FRAMES:
            vid.release()
            out.release()
    print('TEST PREDICTIONS COMPLETE') 
    return [bboxes, classes, sscores]

def inference_images(DIR_TEST, model, OUT_DIR, detection_threshold, CLASSES):
    imagePath = glob.glob(f"{DIR_TEST}/*.png")
    image_extensions = ['jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
    all_extensions = image_extensions + [ext.upper() for ext in image_extensions]  # Add uppercase versions
    for extension in all_extensions:
            imagePath.extend(glob.glob(f"{DIR_TEST}/*.{extension}"))
    all_images = [image_path.split('/')[-1] for image_path in imagePath]
    all_images = sorted(all_images)
    num_images = len(all_images)
    classes = [None] * num_images
    bboxes = [None] * num_images
    sscores = [None] * num_images
    
    for idx, el in enumerate(all_images):
        
        orig_image = cv2.imread(DIR_TEST + '/' + el)
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(float)
        # convert to tensor
        if torch.cuda.is_available():
            image = torch.tensor(image, dtype=torch.float).cuda()
        else:
            image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            sscores[idx] = scores[scores >= detection_threshold]
            
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            bboxes[idx] = boxes
            draw_boxes = bboxes[idx].copy() 
             
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            pred_classes = np.array(pred_classes)
            pred_classes = pred_classes[scores >= detection_threshold]
            classes[idx] = pred_classes
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, str(pred_classes[j]), 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                            2, lineType=cv2.LINE_AA)
            cv2.imwrite(OUT_DIR + '/' + el, orig_image) #The 'el' filepath is broken right now (TODO: FIX) 

        print(f"Image {idx+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE') 
    return [bboxes, classes, sscores]