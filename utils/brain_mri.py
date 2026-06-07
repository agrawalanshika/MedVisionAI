import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]
    


def load_tumor_model():

    model = models.densenet121(
        weights=None
    )

    num_features = (
        model.classifier.in_features
    )

    model.classifier = nn.Linear(
        num_features,
        4
    )

    model.load_state_dict(
        torch.load(
            "models/brain_mri_densenet121.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model
MODEL= load_tumor_model()
TARGET_LAYER = MODEL.features
for module in MODEL.modules():

    if isinstance(
        module,
        torch.nn.ReLU
    ):
        module.inplace = False
def predict_tumor(image):

    

    transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
        ),

        transforms.Resize(
            (224,224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )

    ])

    image =image.convert("RGB")

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