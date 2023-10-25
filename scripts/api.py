from fastapi import FastAPI, Body

from modules.api.models import *
from modules.api import api
import gradio as gr

import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from scripts.vit_model import vit_base_patch16_224_in21k as create_model

cls_model = None
is_load = False
def get_model(device):
    global cls_model, is_load
    if not is_load:
        model_weight_path = "/tmp/models/stable-diffusion/flip.pth"
        cls_model = create_model(num_classes=2, has_logits=False).to(device)
        cls_model.load_state_dict(torch.load(model_weight_path, map_location=device))
        is_load = True
    return cls_model

def flip_monster_api(_: gr.Blocks, app: FastAPI):
    @app.post("/flip_monster")
    async def face_crop(
        input_image: str = Body("", title='flip monster'),
        model: str = Body("buffalo_l", title='face Recognition model'), 
    ):
        print('start')
        image = api.decode_base64_to_image(input_image)

        data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        img = data_transform(image).unsqueeze(0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model weights
        model = get_model(device)
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print(predict_cla)
        if str(predict_cla) == "0":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return {"images": [api.encode_pil_to_base64(image).decode("utf-8")]}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(flip_monster_api)
except:
    pass


# from fastapi import FastAPI, Body

# from modules.api.models import *
# from modules.api import api
# import gradio as gr

# import cv2
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# from scripts.vit_model import vit_base_patch16_224_in21k as create_model


# cls_model = None
# is_load = False
# def get_model(device):
#     global cls_model, is_load
#     if not is_load:
#         model_weight_path = "best_model.pth"
#         cls_model = create_model(num_classes=2, has_logits=False).to(device)
#         cls_model = load_state_dict(torch.load(model_weight_path, map_location=device))
#     return cls_model

# def flip_monster_api(_: gr.Blocks, app: FastAPI):
#     @app.post("/flip_monster")
#     async def flip_monster(
#         input_image: str = Body("", title='flip monster'),
#     ):
#         print('start')
#         input_image = api.decode_base64_to_image(input_image)

#         data_transform = transforms.Compose([transforms.Resize([224, 224]),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

#         img = data_transform(input_image)

#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # load model weights
#         model = get_model(device)
#         model.eval()
#         with torch.no_grad():
#             # predict class
#             output = torch.squeeze(model(img.to(device))).cpu()
#             predict = torch.softmax(output, dim=0)
#             predict_cla = torch.argmax(predict).numpy()

#         if str(predict_cla) == "0":
#             image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
#         else:
#             image = input_image

#         return {"images": [api.encode_pil_to_base64(image).decode("utf-8")]}

# try:
#     import modules.script_callbacks as script_callbacks

#     script_callbacks.on_app_started(flip_monster_api)
# except:
#     print("Something wrong.")
#     pass