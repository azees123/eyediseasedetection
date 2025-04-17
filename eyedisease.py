import numpy as np
from PIL import Image
import tensorflow.lite as tflite
from kivy.resources import resource_find

from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from plyer import filechooser

# Disease label mapping
disease_classes = {
    0: 'Central Serous Chorioretinopathy',
    1: 'Diabetic Retinopathy',
    2: 'Disc Edema',
    3: 'Glaucoma',
    4: 'Healthy',
    5: 'Macular Scar',
    6: 'Myopia',
    7: 'Pterygium',
    8: 'Retinal Detachment',
    9: 'Retinitis Pigmentosa'
}

# TensorFlow Lite model prediction
def predict_image(image_path):
    # Load TFLite model
    model_path = resource_find("eye_disease_model.tflite")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare image
    img = Image.open(image_path).resize((64, 64)).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction)

    return disease_classes.get(predicted_class, 'Unknown')


class EyeDiseaseApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        self.image_display = KivyImage()
        self.result_label = Label(text="Upload an eye image for prediction", font_size=18)
        self.select_button = Button(text='Choose Image', size_hint=(1, 0.2))
        self.select_button.bind(on_release=self.choose_file)

        self.layout.add_widget(self.image_display)
        self.layout.add_widget(self.result_label)
        self.layout.add_widget(self.select_button)

        return self.layout

    def choose_file(self, instance):
        filechooser.open_file(on_selection=self.selected)

    def selected(self, selection):
        if selection:
            image_path = selection[0]
            self.image_display.source = image_path
            disease = predict_image(image_path)
            self.result_label.text = f'Detected: {disease}'
        else:
            popup = Popup(title='Error', content=Label(text='No file selected'), size_hint=(0.6, 0.4))
            popup.open()


if __name__ == '__main__':
    EyeDiseaseApp().run()
