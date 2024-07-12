import panel as pn
import ipywidgets
pn.extension('mathjax')
pn.extension('ipywidgets')
from library import *
from panel.widgets import Tqdm
targets = ipywidgets.Text(
    value='Frog',
    placeholder='Classes:',
    description='Classes:',
    disabled=False   
)
tqdm = Tqdm()
fig = plt.figure()
train_loss_mpl = pn.pane.Matplotlib(fig)

text1 = pn.pane.Markdown("""Before training, split your data into some training images and some validation images. To do this, add each training image as well as its corresponding annotation .xml file into the images/train folder in the FRCNN2 folder. Similarly, add each validation image as well as its corresponding annotation .xml file into the images/test folder. A 90% training image, 10% validation image split is recommended""", width=600)


#def cb(test):
#    return init_annotations(test.replace(" ", "").split(','))
BATCH_SIZE = pn.widgets.IntInput(name='Batch Size', value=4, step=1, start=1, end=16)
RESIZE_TO = pn.widgets.IntInput(name='Downsampling Size', value=512, step=1, start=64, end=1024)
NUM_EPOCHS = pn.widgets.IntInput(name='Number of Epochs', value=20, step=1, start=1, end=1000)
SAVE_PLOTS_EPOCH = pn.widgets.IntInput(name='Save Plots After ___ Epochs', value=1, step=1, start=1, end=1000)
SAVE_MODEL_EPOCH = pn.widgets.IntInput(name='Save Models After ___ Epochs', value=5, step=1, start=1, end=1000)

load_button = pn.widgets.Button(name='Load Data', button_type='primary')

def handle_clicks_load(clicks):
    if clicks > 0:
        train_dataset = getDataset(TRAIN_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), targets.value.replace(" ","").split(','), get_train_transform())
        valid_dataset = getDataset(VALID_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), targets.value.replace(" ","").split(','), get_valid_transform())
        
        return (f"Number of training samples: {len(train_dataset)}\n" + f"Number of validation samples: {len(valid_dataset)}\n")
        
loadOutput = pn.Column(pn.bind(handle_clicks_load, load_button.param.clicks))

NUM_INDEX = pn.widgets.IntInput(name='Sample Index to Visualize', value=0, step=1, start=0, end=1000)
vis_button = pn.widgets.Button(name='Visualize Samples', button_type='primary')

def handle_clicks_vis(clicks):
    if clicks > 0:
        return visualize_sample(TRAIN_DIR, int(RESIZE_TO.value), targets.value.replace(" ","").split(','), NUM_INDEX.value)
        
visOutput = pn.pane.Matplotlib(pn.bind(handle_clicks_vis, vis_button.param.clicks))

newModelName = ipywidgets.Text(
    value='model',
    placeholder='Model Name',
    description='New Model Name',
    disabled=False   
)

train_model_button = pn.widgets.Button(name='Train Model', button_type='primary')

def handle_clicks_train_model(clicks):
    if clicks > 0:
        train_dataset = getDataset(TRAIN_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), targets.value.replace(" ","").split(','), get_train_transform())
        valid_dataset = getDataset(VALID_DIR, int(RESIZE_TO.value), int(RESIZE_TO.value), targets.value.replace(" ","").split(','), get_valid_transform())
        [train_loader, valid_loader] = get_loaders(train_dataset, valid_dataset, int(BATCH_SIZE.value), collate_fn)
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = create_model(num_classes=(len(targets.value.replace(" ","").split(',')) + 1))
        train_model(model, train_loader, valid_loader, DEVICE, newModelName.value,
                                                   int(NUM_EPOCHS.value), MODEL_DIR, PLOT_DIR, int(SAVE_MODEL_EPOCH.value), 
                                                   int(SAVE_PLOTS_EPOCH.value), tqdm(range(0,NUM_EPOCHS.value)),
                                                   train_loss_mpl)
        return "Training Completed"

tModelOutput = pn.Column(pn.bind(handle_clicks_train_model, train_model_button.param.clicks))

output = pn.Column(text1, loadOutput, visOutput, tqdm, train_loss_mpl, tModelOutput)


side = pn.Column(targets, BATCH_SIZE, RESIZE_TO, NUM_EPOCHS, SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH, load_button, NUM_INDEX, vis_button,
                newModelName, train_model_button)
#pn.panel(output)
pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Training",
    sidebar=[side],
    main=[output],
).servable(); 