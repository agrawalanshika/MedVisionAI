import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import transforms


def generate_gradcam(
    image,
    model,
    target_layer
):

    gradients = []
    activations = []

    def forward_hook(
        module,
        input,
        output
    ):
        # Clone to detach from computation graph
        activations.append(
            output.clone().detach()
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        # ✅ .clone() is the critical fix —
        # prevents "view being modified inplace" error
        gradients.append(
            grad_output[0].clone().detach()
        )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_backward_hook(
            backward_hook
        )
    )

    transform = transforms.Compose([

        transforms.Grayscale(
            num_output_channels=3
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

    original_image = image.convert(
        "RGB"
    )

    input_tensor = transform(
        original_image
    )

    input_tensor = (
        input_tensor.unsqueeze(0)
    )

    # Ensure gradients flow through input
    input_tensor.requires_grad_(True)

    model.zero_grad()

    output = model(
        input_tensor
    )

    pred_class = (
        output.argmax(
            dim=1
        ).item()
    )

    output[
        0,
        pred_class
    ].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * acts
    ).sum(
        dim=1
    ).squeeze()

    cam = torch.relu(cam)

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        + 1e-8
    )

    cam = cam.detach().numpy()

    cam = cv2.resize(
        cam,
        original_image.size
    )

    heatmap = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB before blending
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    original_np = np.array(
        original_image
    )

    overlay = cv2.addWeighted(
        original_np,
        0.6,
        heatmap,
        0.4,
        0
    )

    forward_handle.remove()
    backward_handle.remove()

    return (
        Image.fromarray(overlay),
        pred_class
    )