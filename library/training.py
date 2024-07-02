import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from torch.utils.data import Subset, DataLoader
from torchvision.ops import box_iou
import numpy as np
from .utils import get_loaders
from .utils import collate_fn

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def load_model_train(model_name, MODEL_DIR, NUM_CLASSES):
    # set the computation device
    modelPath = './models/' + model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(
        modelPath, map_location=device
    ))
    return model

def train(train_data_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE):
    print('Training')
    #global train_itr
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list

# function for running validation iterations
def validate(valid_data_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE):
    print('Validating')
    #global val_itr
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def train_model(model, train_loader, valid_loader, DEVICE, MODEL_NAME, NUM_EPOCHS, OUT_DIR, PLOT_DIR, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH):
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_list, train_loss_hist, train_itr, DEVICE)
        val_loss = validate(valid_loader, model, val_loss_list, val_loss_hist, val_itr, DEVICE)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
    
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')
    
        if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{PLOT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{PLOT_DIR}/valid_loss_{epoch+1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
    
        plt.close('all')
    return [train_loss_list, val_loss_list]

def get_subsampled_dataset(full_dataset, num_samples):
    """ Randomly subsample the dataset to the specified number of samples. """
    indices = np.random.permutation(len(full_dataset))[:num_samples]
    return Subset(full_dataset, indices)

def validate_mAP(data_loader, model, device):
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    all_scores = []
    all_gt_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Predictions
                pred_boxes = output['boxes'].to('cpu')
                pred_labels = output['labels'].to('cpu')
                scores = output['scores'].to('cpu')

                # Ground truth
                gt_boxes = targets[i]['boxes']
                gt_labels = targets[i]['labels']

                all_gt_boxes.append(gt_boxes)
                all_pred_boxes.append(pred_boxes)
                all_scores.append(scores)
                all_gt_labels.append(gt_labels)
                all_pred_labels.append(pred_labels)

    # Calculate mAP
    return calculate_mAP(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_scores)

def calculate_mAP(all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_scores, iou_threshold=0.5):
    # This function computes the mAP at the specified IoU threshold
    num_classes = max(max(l) for l in all_gt_labels) + 1  # assuming labels are zero-indexed

    average_precisions = []
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Collect data for this class
        for i in range(len(all_gt_labels)):
            gt_indices = (all_gt_labels[i] == c)
            pred_indices = (all_pred_labels[i] == c)

            if gt_indices.any():
                ground_truths.extend(all_gt_boxes[i][gt_indices].tolist())

            if pred_indices.any():
                detections.extend(
                    zip(all_pred_boxes[i][pred_indices].tolist(),
                        all_scores[i][pred_indices].tolist())
                )

        # Sort by scores
        detections.sort(key=lambda x: x[1], reverse=True)
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        detected_boxes = []

        # Calculate TP and FP
        for d_idx, detection in enumerate(detections):
            pred_box, score = detection
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt_box in enumerate(ground_truths):
                iou = box_iou(torch.tensor([pred_box]), torch.tensor([gt_box])).item()
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_threshold:
                if max_gt_idx not in detected_boxes:
                    TP[d_idx] = 1
                    detected_boxes.append(max_gt_idx)
                else:
                    FP[d_idx] = 1
            else:
                FP[d_idx] = 1

        # Compute Precision and Recall
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recalls = TP_cumsum / (len(ground_truths) + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

        # Compute AP
        ap = np.trapz(precisions, recalls)
        average_precisions.append(ap)

    # Mean AP across all classes
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP

def run_experiment(full_train_dataset, valid_dataset, num_classes, BATCH_SIZE, NUM_EXPERIMENTS=10, EPOCHS_PER_EXPERIMENT=20, TRIALS_PER_EXPERIMENT=3):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directory setup for models and plots
    model_dir = 'models/'
    plot_dir = 'plots/'

    # Calculate number of samples for each experiment
    total_samples = len(full_train_dataset)
    split_sizes = np.linspace(0, total_samples, NUM_EXPERIMENTS + 1, dtype=int)[1:]

    results = []

    # Create plot
    plt.figure()
    mean_mAPs = []
    std_mAPs = []

    for num_samples in split_sizes:
        mAPs = []
        for trial in range(TRIALS_PER_EXPERIMENT):
            print(f"\nRunning training with {num_samples} samples, trial {trial + 1}...")

            # Subsample the training dataset
            train_subset = get_subsampled_dataset(full_train_dataset, num_samples)
            train_loader, valid_loader = get_loaders(train_subset, valid_dataset, BATCH_SIZE, collate_fn)

            # Initialize a fresh instance of the model
            model = create_model(num_classes).to(device)

            # Train the model
            train_model(model, train_loader, valid_loader, device, 'experiment_model', EPOCHS_PER_EXPERIMENT, model_dir, plot_dir, 5, 5)

            # Evaluate the model
            val_mAP = validate_mAP(valid_loader, model, device)
            mAPs.append(val_mAP)
            print(f"Trial {trial + 1}: Validation mAP = {val_mAP:.3f}")

        # Compute statistics
        mean_mAP = np.mean(mAPs)
        std_mAP = np.std(mAPs)
        mean_mAPs.append(mean_mAP)
        std_mAPs.append(std_mAP)
        results.append((num_samples, mean_mAP, std_mAP))
        print(f"Finished {num_samples} samples: Mean Validation mAP = {mean_mAP:.3f}, Std Dev = {std_mAP:.3f}")

    # Plotting results
    plt.errorbar(split_sizes, mean_mAPs, yerr=std_mAPs, fmt='-o', capsize=5)
    plt.title('Mean and Standard Deviation of Validation mAP')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Validation mAP')
    plt.grid(True)
    plt.savefig(f"{plot_dir}/mAP_results.png")
    plt.show()

    return results