from library import *
import panel as pn
import ipywidgets
from panel.widgets import Tqdm
pn.extension('ipywidgets')
tqdm = Tqdm()

threshold = pn.widgets.FloatInput(name='FloatInput', value=0.9, step=1e-2, start=0, end=1)
model_name = ipywidgets.Text(
    value='model20.pth',
    placeholder='Model Name:',
    description='Model Name:',
    disabled=False   
)
classes_text = ipywidgets.Text(
    value='frog',
    placeholder='Classes:',
    description='Classes:',
    disabled=False   
)
folder_name = ipywidgets.Text(
    value='./test_data/test_images/',
    placeholder='Image Folder:',
    description='Image Folder:',
    disabled=False   
)

inference_button = pn.widgets.Button(name='Inference Images', button_type='primary')

def handle_inf_images(clicks):
    if clicks > 0:
        model = load_model(model_name.value, MODEL_DIR, 2)
        [boxFileName, classFileName, scoreFileName] = ['boxes', 'classes', 'scores']
        print(['background'] + classes_text.value.replace(" ","").split(','))
        [bboxes, classes, scores] = inference_images(folder_name.value, model, OUT_DIR, threshold.value, ['background'] + classes_text.value.replace(" ","").split(','), tqdm)
        saveBoxesClassesScores(boxFileName, classFileName, scoreFileName, bboxes, classes, scores, OUT_DIR)
        
te = pn.Column(pn.bind(handle_inf_images, inference_button.param.clicks))


#folderName = './test_data/test_images/'
#[boxFileName, classFileName, scoreFileName] = ['boxes', 'classes_text', 'scores']
#[bboxes, classes, scores] = inference_images(folderName, model, OUT_DIR, detection_threshold, CLASSES)
#saveBoxesClassesScores(boxFileName, classFileName, scoreFileName, bboxes, classes, scores, OUT_DIR)


side = pn.Column(threshold, model_name, classes_text, folder_name, inference_button)

pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Inferencing",
    sidebar=[side],
    main=[tqdm, te],
).servable(); 