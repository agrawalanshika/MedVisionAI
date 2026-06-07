import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image


CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


def load_pneumonia_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "models/resnet50_pneumonia_v2.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# Load model only once
MODEL = load_pneumonia_model()
TARGET_LAYER = MODEL.layer4
for module in MODEL.modules():

    if isinstance(
        module,
        torch.nn.ReLU
    ):
        module.inplace = False


def predict_pneumonia(image):

    transform = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.ToTensor()

    ])

    image = image.convert(
        "RGB"
    )

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

        probabilities = torch.softmax(
            output,
            dim=1
        )

        pred_class = probabilities.argmax(
            dim=1
        ).item()

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