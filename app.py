# imports and class_names
import gradio as gr
import os
import torch

from model import create_effnetb2_model 
from timer import default_timer as timer
from typing import Tuple, Dict

#setup class names 
with open('class_names.txt', 'r') as f:
  class_names = [food_name.strip() for food_name in f.readlines()]

#model and transforms 
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

#laod weights
effnetb2.load_state_dict(
    torch.load(f='09_pretrained_effnetb2_feature_extractor_food101_big.pth',
               map_location=torch.device('cpu'))
)
 
# pred fn
def predict(img) -> Tuple[Dict, float]:

  start_time = timer()

  #transform image
  img = effnetb2_transforms(img).unsqueeze(0)

  #put model in eval mode and make prediction
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb2(img), dim=1)

  #create a pred label and pred probability dict
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  #calc pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)
  return pred_labels_and_probs, pred_time

#Gradio app
title = 'FoodVision Big-V 🍴'
description = "An EfficientNetB2 feature extractor computer vision model to classify images 101 classes of food from the Food101 dataset."

#create example list
example_list = [['examples/' + example] for example in os.listdir('examples')]

demo = gr.Interface(inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=5, label='Predictions'),
                             gr.Number(label='Prediction time (s)')],
                    examples=example_list,
                    fn=predict,
                    title=title,
                    description=description)

demo.launch(debug=False,
            share=True)
