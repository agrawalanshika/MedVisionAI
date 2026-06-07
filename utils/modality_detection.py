import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0
)


CLASS_NAMES = [
    "brain_ct",
    "brain_mri",
    "chest_ct",
    "chest_xray"
]


def load_modality_model():

    model = efficientnet_b0(
        weights=None
    )

    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.2,
            inplace=True
        ),

        nn.Linear(
            1280,
            4
        )

    )

    model.load_state_dict(

        torch.load(
            "models/modality_classifier.pth",
            map_location="cpu"
        )

    )

    model.eval()

    return model

MODEL = load_modality_model()

def predict_modality(image):


    transform = transforms.Compose([

        transforms.Lambda(
            lambda img: img.convert("RGB")
        ),

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]

        )

    ])


    image = transform(
        image
    )

    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = MODEL(
            image
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        pred = probabilities.argmax(
            dim=1
        ).item()

    return {

        "modality":
            CLASS_NAMES[pred],

        "confidence":
            probabilities[0][pred].item()
            * 100,

        "probabilities": {

            CLASS_NAMES[i]:
            probabilities[0][i].item()
            * 100

            for i in range(
                len(CLASS_NAMES)
            )

        }

    }