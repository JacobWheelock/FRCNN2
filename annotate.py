from library import init_annotations, split_images_and_xml
import panel as pn
import ipywidgets
pn.extension('ipywidgets')
classes = ['frog']
textbar = ipywidgets.Text(
    value='Frog',
    placeholder='Type something',
    description='Classes:',
    disabled=False   
)
def cb(test):
    return init_annotations(test.replace(" ", "").split(','))
x1 = pn.Row(pn.bind(cb,textbar))
pn.panel(x1)

test_ratio = pn.widgets.FloatInput(name='Test Ratio', value=0.1, step=1e-1, start=0, end=1)
shuffle_button = pn.widgets.Button(name='Shuffle Data', button_type='primary')
source_folder = './annotations/'
#shuffle_button.on_click(split_images_and_xml(source_folder=source_folder, test_ratio=float(test_ratio.value)))
#def cb2(value):
#    split_images_and_xml(source_folder=source_folder, test_ratio=float(test_ratio.value))
#x2 = pn.Row(pn.bind(cb2, shuffle_button))
def handle_clicks(clicks):
    if clicks > 0:
        split_images_and_xml(source_folder=source_folder, test_ratio=float(test_ratio.value))
but = pn.Column(shuffle_button,pn.bind(handle_clicks, shuffle_button.param.clicks))
side = pn.Column(textbar, test_ratio, but)

pn.template.MaterialTemplate(
    site="EZ-FRCNN",
    title="Annotator",
    sidebar=[side],
    main=[x1],
).servable(); 