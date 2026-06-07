import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image


CLASS_NAMES = [
    "COVID",
    "non-COVID"
]


def load_covid_model():

    model = models.convnext_tiny(
        weights=None
    )

    model.classifier[2] = nn.Linear(
        in_features=768,
        out_features=2
    )

    model.load_state_dict(
        torch.load(
            "models/chest_ct_convnext_tiny.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# Load once
MODEL = load_covid_model()
TARGET_LAYER = MODEL.features[-2]

def predict_covid_ct(image):

    transform = transforms.Compose([

        transforms.Lambda(
            lambda img:
            img.convert("RGB")
        ),

        transforms.Resize(
            (224,224)
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

    image = image.unsqueeze(
        0
    )

    with torch.no_grad():

        output = MODEL(
            image
        )

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )
        )

        pred_class = (
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = (
            probabilities[
                0,
                pred_class
            ].item()
            * 100
        )

    return {

        "prediction":
            CLASS_NAMES[
                pred_class
            ],

        "confidence":
            confidence,

        "probabilities": {

            CLASS_NAMES[i]:
            probabilities[
                0,
                i
            ].item() * 100

            for i in range(
                len(
                    CLASS_NAMES
                )
            )

        }

    }