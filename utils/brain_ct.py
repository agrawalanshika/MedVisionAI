import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0
)

from torchvision import transforms

from PIL import Image


CLASS_NAMES = [
    "Normal",
    "Hemorrhage"
]


def load_hemorrhage_model():

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
            2
        )

    )

    model.load_state_dict(
        torch.load(
            "models/brain_ct_hemorrhage.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# Load only once
MODEL = load_hemorrhage_model()
TARGET_LAYER = MODEL.features[-1]
for module in MODEL.modules():

    if isinstance(
        module,
        torch.nn.ReLU
    ):
        module.inplace = False


def predict_hemorrhage(image):

    transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (224,224)
        ),

        transforms.ToTensor()

    ])

    image = image.convert("L")

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