from utils.chest_xray import (
    predict_pneumonia
)

from utils.brain_mri import (
    predict_tumor
)

from utils.chest_ct import (
    predict_covid_ct
)

from utils.brain_ct import (
    predict_hemorrhage
)


def route_prediction(
    image,
    modality
):

    if modality == "chest_xray":

        return predict_pneumonia(
            image
        )

    elif modality == "brain_mri":

        return predict_tumor(
            image
        )

    elif modality == "chest_ct":

        return predict_covid_ct(
            image
        )

    elif modality == "brain_ct":

        return predict_hemorrhage(
            image
        )

    else:

        raise ValueError(
            f"Unknown modality: {modality}"
        )