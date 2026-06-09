import streamlit as st
import pydicom
import numpy as np
import time

from PIL import Image

from utils.modality_detection import (
    predict_modality
)

from utils.router import (
    route_prediction
)

from utils.gradcam import (
    generate_gradcam
)

from utils.brain_mri import (
    MODEL as MRI_MODEL,
    TARGET_LAYER as MRI_LAYER
)

from utils.brain_ct import (
    MODEL as CT_MODEL,
    TARGET_LAYER as CT_LAYER
)

from utils.chest_xray import (
    MODEL as XRAY_MODEL,
    TARGET_LAYER as XRAY_LAYER
)

from utils.chest_ct import (
    MODEL as COVID_MODEL,
    TARGET_LAYER as COVID_LAYER
)

from utils.report_generator import (
    generate_report
)


st.set_page_config(
    page_title="MedVisionAI",
    page_icon="🩺",
    layout="wide"
)


def load_image(uploaded_file):

    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext in [
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]:

        image = Image.open(uploaded_file)
        return image

    elif file_ext == "dcm":

        dicom = pydicom.dcmread(uploaded_file)
        image = dicom.pixel_array

        if len(image.shape) == 3:
            if image.shape[-1] in [3, 4]:
                pass
            else:
                middle_slice = image.shape[0] // 2
                image = image[middle_slice]

        image = image.astype(np.float32)

        if hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                image = image.max() - image

        image = (
            image - image.min()
        ) / (
            image.max() - image.min() + 1e-8
        )

        image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = Image.fromarray(image)
        elif len(image.shape) == 3:
            image = Image.fromarray(image)
        else:
            raise ValueError(
                f"Unsupported DICOM shape: {image.shape}"
            )

        image = image.convert("RGB")
        return image

    else:
        raise ValueError("Unsupported file format")


def validate_image(image):
    """
    Validate image suitability for single-scan analysis.
    Returns (is_valid, reason).
    """

    w, h = image.size
    aspect_ratio = w / h

    if aspect_ratio > 1.8:
        return False, (
            "This image appears to contain multiple scan frames "
            "side by side. Please upload a single scan image "
            "without composite panels. "
            "Multi-frame or annotated research figures are not "
            "supported for automated analysis."
        )

    if aspect_ratio < 0.3:
        return False, (
            "This image has an unusually tall aspect ratio. "
            "Please upload a standard single medical scan."
        )

    if w < 100 or h < 100:
        return False, (
            "Image resolution is too low for reliable analysis. "
            "Please upload a higher quality scan "
            "with a minimum resolution of 100×100 pixels."
        )

    img_array = np.array(image.convert("RGB"))
    r = img_array[:, :, 0].astype(np.float32)
    g = img_array[:, :, 1].astype(np.float32)
    b = img_array[:, :, 2].astype(np.float32)

    yellow_mask = (r > 180) & (g > 180) & (b < 80)
    yellow_ratio = yellow_mask.sum() / (w * h)

    if yellow_ratio > 0.005:
        return False, (
            "This image appears to contain annotation overlays "
            "(e.g. bounding boxes or markings). "
            "Please upload the original unmodified scan without "
            "any drawn annotations, as overlays interfere with "
            "the AI analysis."
        )

    return True, None


def format_modality(modality):

    if modality is None:
        return "Unknown"

    mapping = {
        "brain_ct":         "Brain CT",
        "brain_mri":        "Brain MRI",
        "chest_ct":         "Chest CT",
        "chest_xray":       "Chest X-Ray",
        "-- Select one --": "-- Select one --",
    }

    return mapping.get(
        modality, modality.replace("_", " ").title()
    )


def format_prediction(prediction):
    """
    Normalize raw model prediction labels to
    clean display-friendly strings.

    Covers all known variants a model might return —
    underscore, bare word, mixed case, with or without
    the word "tumor" appended.
    """

    if prediction is None:
        return "Unknown"

    mapping = {
        # ── Chest X-Ray ───────────────────────────────
        "PNEUMONIA":            "Pneumonia",
        "pneumonia":            "Pneumonia",
        "NORMAL":               "Normal",
        "normal":               "Normal",

        # ── Chest CT ──────────────────────────────────
        "COVID":                "Covid",
        "Covid":                "Covid",
        "covid":                "Covid",
        "non-COVID":            "Non-Covid",
        "Non-COVID":            "Non-Covid",
        "NON-COVID":            "Non-Covid",
        "non_COVID":            "Non-Covid",
        "noncovid":             "Non-Covid",
        "non-covid":            "Non-Covid",
        "non covid":            "Non-Covid",

        # ── Brain MRI — with underscore (raw labels) ──
        "glioma_tumor":         "Glioma Tumor",
        "meningioma_tumor":     "Meningioma Tumor",
        "pituitary_tumor":      "Pituitary Tumor",
        "no_tumor":             "No Tumor",

        # ── Brain MRI — bare word (model returns name only)
        "Glioma":               "Glioma Tumor",
        "glioma":               "Glioma Tumor",
        "Meningioma":           "Meningioma Tumor",
        "meningioma":           "Meningioma Tumor",
        "Pituitary":            "Pituitary Tumor",
        "pituitary":            "Pituitary Tumor",
        "notumor":              "No Tumor",
        "no tumor":             "No Tumor",
        "No tumor":             "No Tumor",

        # ── Brain MRI — already formatted (idempotent) ─
        "Glioma Tumor":         "Glioma Tumor",
        "Meningioma Tumor":     "Meningioma Tumor",
        "Pituitary Tumor":      "Pituitary Tumor",
        "No Tumor":             "No Tumor",

        # ── Brain CT ──────────────────────────────────
        "Hemorrhage":           "Hemorrhage",
        "hemorrhage":           "Hemorrhage",
        "Normal":               "Normal",
    }

    return mapping.get(prediction, prediction)


def get_gradcam_components(modality):

    if modality == "brain_mri":
        return MRI_MODEL, MRI_LAYER
    elif modality == "brain_ct":
        return CT_MODEL, CT_LAYER
    elif modality == "chest_xray":
        return XRAY_MODEL, XRAY_LAYER
    elif modality == "chest_ct":
        return COVID_MODEL, COVID_LAYER

    return None, None


# Tumor prediction labels after formatting —
# used in get_disease_context for reliable matching
TUMOR_PREDICTIONS = {
    "glioma tumor",
    "meningioma tumor",
    "pituitary tumor"
}

NEGATIVE_PREDICTIONS = {
    "no tumor",
    "non-covid",
    "non_covid",
    "non covid",
    "normal"
}


def get_disease_context(prediction, confidence):
    """Return (type, message) based on prediction + confidence."""

    # Always use lowercase for matching
    prediction_lower = prediction.lower()

    # ── Low confidence ─────────────────────────────────
    if confidence < 70:
        return (
            "warning",
            f"**Inconclusive Result — {prediction}** · "
            f"Confidence: {confidence:.1f}% · "
            f"The model's confidence is below the reliable threshold. "
            f"This result should not be used for any clinical decision. "
            f"Independent review by a qualified radiologist is required "
            f"before any action is taken."
        )

    # ── Tumor findings (Brain MRI) ─────────────────────
    # Match against the set of known formatted tumor names
    if prediction_lower in TUMOR_PREDICTIONS:
        return (
            "error",
            f"**Finding: {prediction}** · "
            f"Confidence: {confidence:.1f}% · "
            f"The model has identified features consistent with "
            f"a {prediction} with high confidence. "
            f"This finding requires urgent review by a qualified "
            f"radiologist or oncologist. "
            f"Do not act on this result without professional confirmation."
        )

    # ── COVID (Chest CT) ───────────────────────────────
    elif (
        "covid" in prediction_lower
        and "non" not in prediction_lower
    ):
        return (
            "error",
            f"**Finding: {prediction}** · "
            f"Confidence: {confidence:.1f}% · "
            f"The model has identified features consistent with COVID-19 "
            f"pneumonia with high confidence. This result must be confirmed "
            f"by a licensed radiologist. Appropriate isolation and clinical "
            f"assessment are advised."
        )

    # ── Pneumonia (Chest X-Ray) ────────────────────────
    elif "pneumonia" in prediction_lower:
        return (
            "warning",
            f"**Finding: {prediction}** · "
            f"Confidence: {confidence:.1f}% · "
            f"The model has identified features consistent with pneumonia. "
            f"Clinical correlation and review by a qualified physician "
            f"is recommended before any treatment decision."
        )

    # ── Negative / normal findings ─────────────────────
    elif prediction_lower in NEGATIVE_PREDICTIONS:
        return (
            "success",
            f"**Finding: {prediction}** · "
            f"Confidence: {confidence:.1f}% · "
            f"No significant pathology detected at this confidence level. "
            f"Routine clinical follow-up as advised by your "
            f"healthcare provider is recommended."
        )

    # ── Fallback ───────────────────────────────────────
    return (
        "info",
        f"**Finding: {prediction}** · "
        f"Confidence: {confidence:.1f}% · "
        f"Please consult a qualified medical professional "
        f"to interpret this result in the appropriate clinical context."
    )


MODALITY_FAMILY = {
    "brain_ct":   ["brain_mri"],
    "brain_mri":  ["brain_ct"],
    "chest_ct":   [],
    "chest_xray": [],
}


def get_correction_options(detected_modality):
    return MODALITY_FAMILY.get(detected_modality, [])


def confidence_bar(confidence):

    if confidence >= 80:
        color = "#22c55e"
    elif confidence >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    st.markdown(
        f"""
        <div style="
            background:#1e1e2e;
            border-radius:6px;
            height:10px;
            width:100%;
            margin-top:4px;
        ">
            <div style="
                width:{confidence}%;
                background:{color};
                height:10px;
                border-radius:6px;
                transition:width 0.4s ease;
            "></div>
        </div>
        <p style="font-size:0.75rem;color:#9ca3af;margin:2px 0 0 0;">
            {confidence:.2f}% confidence
        </p>
        """,
        unsafe_allow_html=True
    )


def on_radio_change():
    for key in [
        "verification_confirmed",
        "corrected_modality_select",
        "corrected_modality_snapshot",
        "verified_answer",
        "disease_result",
        "gradcam_image",
        "analysis_run"
    ]:
        if key in st.session_state:
            del st.session_state[key]


def on_dropdown_change():
    for key in [
        "verification_confirmed",
        "verified_answer",
        "disease_result",
        "gradcam_image",
        "analysis_run"
    ]:
        if key in st.session_state:
            del st.session_state[key]


st.title("🩺 MedVisionAI - A Multimodal Medical Imaging Platform")

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "dcm"
    ]
)

if uploaded_file:

    try:

        image = load_image(uploaded_file)

        # ── Validate before anything else ─────────────
        is_valid, reason = validate_image(image)

        if not is_valid:
            st.error(
                f"⚠️ **Image Not Accepted** — {reason}"
            )
            st.stop()

        # ── Toast only on fresh upload ─────────────────
        if (
            "last_uploaded" not in st.session_state
            or st.session_state.last_uploaded != uploaded_file.name
        ):
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state["just_uploaded"] = True

            for key in [
                "verification_radio",
                "verification_confirmed",
                "corrected_modality_select",
                "corrected_modality_snapshot",
                "verified_answer",
                "disease_result",
                "gradcam_image",
                "analysis_run",
                "modality_result"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

        if st.session_state.pop("just_uploaded", False):
            st.toast(
                "Image loaded successfully!",
                icon="🖼️"
            )

        if "modality_result" not in st.session_state:

            with st.spinner(
                "Detecting imaging modality..."
            ):
                time.sleep(1.5)
                result = predict_modality(image)

            st.session_state["modality_result"] = result

        modality = st.session_state["modality_result"]["modality"]
        confidence = st.session_state["modality_result"]["confidence"]

        with st.container(border=True):

            left_col, right_col = st.columns([1, 2])

            with left_col:

                st.image(
                    image,
                    use_container_width=True
                )

                st.caption(
                    f"📄 {uploaded_file.name}"
                )

                

            with right_col:

                radio_selection = st.session_state.get(
                    "verification_radio", None
                )

                verification_confirmed = st.session_state.get(
                    "verification_confirmed", False
                )

                selected_from_dropdown = st.session_state.get(
                    "corrected_modality_snapshot", None
                ) if verification_confirmed else st.session_state.get(
                    "corrected_modality_select", None
                )

                analysis_run = st.session_state.get(
                    "analysis_run", False
                )

                st.markdown(
                    f"### Detected Modality — "
                    f"{format_modality(modality)}"
                )

                # ═══════════════════════════════════════
                # PRE-CONFIRM
                # ═══════════════════════════════════════
                if not verification_confirmed:

                    st.markdown(
                        "Please confirm the detected modality below."
                    )

                    st.radio(
                        "Please verify",
                        ["Yes", "No"],
                        index=None,
                        horizontal=True,
                        key="verification_radio",
                        on_change=on_radio_change,
                        label_visibility="collapsed"
                    )

                    radio_selection = st.session_state.get(
                        "verification_radio", None
                    )

                    if radio_selection == "No":

                        correction_options = get_correction_options(
                            modality
                        )

                        if correction_options:

                            st.selectbox(
                                "Select Correct Modality",
                                ["-- Select one --"] + correction_options,
                                format_func=lambda x: (
                                    x if x == "-- Select one --"
                                    else format_modality(x)
                                ),
                                key="corrected_modality_select",
                                on_change=on_dropdown_change
                            )

                            selected_from_dropdown = st.session_state.get(
                                "corrected_modality_select", None
                            )

                        else:

                            st.error(
                                f"⚠️ **Modality Correction Unavailable** — "
                                f"The system has detected this image as "
                                f"**{format_modality(modality)}** with "
                                f"**{confidence:.1f}% confidence**. "
                                f"If you believe this classification is "
                                f"incorrect, please consult a qualified "
                                f"radiologist or imaging specialist before "
                                f"proceeding. Cross-modality corrections "
                                f"for this scan type are not permitted "
                                f"without clinical verification."
                            )

                            selected_from_dropdown = None

                    if radio_selection is not None:

                        correction_options = get_correction_options(
                            modality
                        )

                        real_selection = (
                            selected_from_dropdown is not None
                            and selected_from_dropdown != "-- Select one --"
                        )

                        no_correction_available = (
                            radio_selection == "No"
                            and not correction_options
                        )

                        dropdown_ready = (
                            radio_selection == "Yes"
                            or (
                                radio_selection == "No"
                                and real_selection
                                and not no_correction_available
                            )
                        )

                        if dropdown_ready:

                            correction_label = (
                                "detected modality is correct"
                                if radio_selection == "Yes"
                                else f"correct modality is "
                                     f"**{format_modality(selected_from_dropdown)}**"
                            )

                            st.markdown(
                                f"**Confirm:** You selected "
                                f"**{radio_selection}** — "
                                f"{correction_label}"
                            )

                            if st.button(
                                "✅ Confirm Verification",
                                type="primary"
                            ):

                                with st.spinner(
                                    "Confirming verification..."
                                ):
                                    time.sleep(0.8)

                                st.session_state[
                                    "verification_confirmed"
                                ] = True

                                if radio_selection == "No":
                                    st.session_state[
                                        "corrected_modality_snapshot"
                                    ] = selected_from_dropdown

                                st.session_state[
                                    "verified_answer"
                                ] = radio_selection

                                st.rerun()

                        elif (
                            radio_selection == "No"
                            and correction_options
                            and not real_selection
                        ):

                            st.info(
                                "⏳ Please select the correct "
                                "modality from the dropdown above."
                            )

                    else:

                        st.info(
                            "⏳ Awaiting your review..."
                        )

                # ═══════════════════════════════════════
                # POST-CONFIRM
                # ═══════════════════════════════════════
                else:

                    verified_answer = st.session_state.get(
                        "verified_answer", None
                    )

                    if verified_answer == "Yes":

                        final_modality = modality

                        st.success(
                            f"✅ Verified — Selected Modality: "
                            f"**{format_modality(final_modality)}**"
                        )

                    elif verified_answer == "No":

                        final_modality = selected_from_dropdown

                        st.info(
                            f"🔄 Corrected Modality: "
                            f"**{format_modality(final_modality)}**"
                        )

                        st.success(
                            f"✅ Corrected — Selected Modality: "
                            f"**{format_modality(final_modality)}**"
                        )

                    else:

                        final_modality = modality

                    if not analysis_run:

                        if st.button(
                            "🔬 Run Disease Analysis",
                            use_container_width=True
                        ):

                            with st.spinner(
                                "Analysing disease..."
                            ):

                                disease_result = route_prediction(
                                    image,
                                    final_modality
                                )

                                disease_result["prediction"] = (
                                    format_prediction(
                                        disease_result["prediction"]
                                    )
                                )

                                disease_result["probabilities"] = {
                                    format_prediction(k): v
                                    for k, v in disease_result[
                                        "probabilities"
                                    ].items()
                                }

                                model, layer = get_gradcam_components(
                                    final_modality
                                )

                                if model is not None:

                                    gradcam_image, _ = generate_gradcam(
                                        image,
                                        model,
                                        layer
                                    )

                                    st.session_state[
                                        "gradcam_image"
                                    ] = gradcam_image

                            st.session_state["disease_result"] = disease_result
                            st.session_state["analysis_run"] = True
                            st.rerun()

                    if "disease_result" in st.session_state:

                        d_result = st.session_state["disease_result"]

                        st.subheader("Clinical Findings")

                        metric1, metric2 = st.columns(2)

                        with metric1:

                            st.metric(
                                "Prediction",
                                d_result["prediction"]
                            )

                        with metric2:

                            st.metric(
                                "Confidence",
                                f"{d_result['confidence']:.2f}%"
                            )

                            confidence_bar(
                                d_result["confidence"]
                            )

                        context_type, context_msg = get_disease_context(
                            d_result["prediction"],
                            d_result["confidence"]
                        )

                        if context_type == "error":
                            st.error(context_msg)

                        elif context_type == "warning":
                            st.warning(context_msg)

                        elif context_type == "success":
                            st.success(context_msg)

                        else:
                            st.info(context_msg)

                        action_col1, action_col2 = st.columns(2)

                        with action_col1:

                            if st.button(
                                "🔄 Re-run Analysis",
                                use_container_width=True
                            ):

                                with st.spinner(
                                    "Analysing disease..."
                                ):

                                    disease_result = route_prediction(
                                        image,
                                        final_modality
                                    )

                                    disease_result["prediction"] = (
                                        format_prediction(
                                            disease_result["prediction"]
                                        )
                                    )

                                    disease_result["probabilities"] = {
                                        format_prediction(k): v
                                        for k, v in disease_result[
                                            "probabilities"
                                        ].items()
                                    }

                                    model, layer = get_gradcam_components(
                                        final_modality
                                    )

                                    if model is not None:

                                        gradcam_image, _ = generate_gradcam(
                                            image,
                                            model,
                                            layer
                                        )

                                        st.session_state[
                                            "gradcam_image"
                                        ] = gradcam_image

                                st.session_state["disease_result"] = disease_result
                                st.session_state["analysis_run"] = True
                                st.rerun()

                        with action_col2:

                            verification_status = (
                                "Verified"
                                if verified_answer == "Yes"
                                else "Corrected"
                            )

                            pdf_path = generate_report(

                                original_image=image,

                                gradcam_image=st.session_state.get(
                                    "gradcam_image", None
                                ),

                                filename=uploaded_file.name,

                                detected_modality=format_modality(
                                    modality
                                ),

                                modality_confidence=confidence,

                                verification_status=verification_status,

                                final_modality=format_modality(
                                    final_modality
                                ),

                                disease_result=d_result
                            )

                            with open(
                                pdf_path,
                                "rb"
                            ) as pdf_file:

                                st.download_button(
                                    label="📄 Download Diagnostic Report",
                                    data=pdf_file,
                                    file_name="MedVisionAI_Report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )

    except Exception as e:

        st.error(
            f"Error loading image: {e}"
        )