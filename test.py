# from utils.chest_xray import predict_pneumonia
# from utils.brain_mri import predict_tumor
# from utils.chest_ct import predict_covid_ct
# from utils.brain_ct import predict_hemorrhage
# from utils.modality_detection import (
#     predict_modality
# )


from utils.modality_detection import (
    predict_modality
)

from utils.router import (
    route_prediction
)

image_path = "test_images/modality detection/chestct2.webp"

modality_result = predict_modality(
    image_path
)

print(
    "Detected:",
    modality_result
)

final_result = route_prediction(

    image_path,

    modality_result[
        "modality"
    ]

)

print(
    "Disease Prediction:"
)

print(
    final_result
)


# from utils.router import (
#     route_prediction
# )

# result = route_prediction(
#     image_path=
#     "test_images/modality detection/brainct2.jpg",

#     modality=
#     "brain_ct"
# )

# print(result)

# result = predict_modality(
#     "test_images/modality detection/mri1.png"
# )

# print(result)

# print(
#     predict_pneumonia(
#         "test_images/pneumonia/pneumonia.png"
#     )
# )

# print(
#     predict_tumor(
#         "test_images/tumor/glioma.jpg"
#     )
# )

# print(
#     predict_covid_ct(
#         "test_images/covid/covid1.png"
#     )
# )

# print(
#     predict_hemorrhage(
#         "test_images/hemorrhage/h1.jpg"
#     )
#)