from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image as RLImage
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from reportlab.lib import colors

from reportlab.lib.units import inch

from datetime import datetime

import tempfile
import os


def generate_report(
    original_image,
    gradcam_image,
    filename,
    detected_modality,
    modality_confidence,
    verification_status,
    final_modality,
    disease_result
):

    timestamp = datetime.now()

    analysis_id = (
        f"MVA-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    )

    pdf_path = os.path.join(
        tempfile.gettempdir(),
        "medvisionai_report.pdf"
    )

    doc = SimpleDocTemplate(
        pdf_path
    )

    styles = getSampleStyleSheet()

    story = []


    story.append(
        Paragraph(
            "MEDVISIONAI DIAGNOSTIC REPORT",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "<b>Analysis Information</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Analysis ID: {analysis_id}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Generated On: {timestamp.strftime('%d %B %Y %H:%M:%S')}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 12)
    )


    story.append(
        Paragraph(
            "<b>Uploaded Image Information</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Filename: {filename}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    preview_temp = os.path.join(
        tempfile.gettempdir(),
        "preview_image.png"
    )

    original_image.save(preview_temp)

    story.append(
        RLImage(
            preview_temp,
            hAlign="LEFT",
            width=2.5 * inch,
            height=2.5 * inch
        )
    )

    story.append(
        Spacer(1, 12)
    )


    story.append(
        Paragraph(
            "<b>Modality Analysis</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Detected Modality: {detected_modality}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Modality Confidence: {modality_confidence:.2f}%",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Verification Status: {verification_status}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Final Selected Modality: {final_modality}",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 12)
    )


    story.append(
        Paragraph(
            "<b>Disease Analysis</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Prediction: {disease_result['prediction']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Confidence: {disease_result['confidence']:.2f}%",
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 12)
    )


    story.append(
        Paragraph(
            "<b>Class Probability Distribution</b>",
            styles["Heading2"]
        )
    )

    for cls, prob in disease_result[
        "probabilities"
    ].items():

        story.append(
            Paragraph(
                f"{cls}: {prob:.2f}%",
                styles["BodyText"]
            )
        )

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Explainability Analysis",
            styles["Heading1"]
        )
    )

    original_temp = os.path.join(
        tempfile.gettempdir(),
        "original_image.png"
    )

    gradcam_temp = os.path.join(
        tempfile.gettempdir(),
        "gradcam_image.png"
    )

    original_image.save(original_temp)

    if gradcam_image is not None:
        gradcam_image.save(gradcam_temp)

    story.append(
        Paragraph(
            "<b>Original Uploaded Image</b>",
            styles["Heading2"]
        )
    )

    story.append(
        RLImage(
            original_temp,
            width=4 * inch,
            height=4 * inch
        )
    )

    story.append(
        Spacer(1, 12)
    )

    story.append(
        Paragraph(
            "<b>Grad-CAM Visualization</b>",
            styles["Heading2"]
        )
    )

    if gradcam_image is not None:

        story.append(
            RLImage(
                gradcam_temp,
                width=4 * inch,
                height=4 * inch
            )
        )

    else:

        story.append(
            Paragraph(
                "Grad-CAM visualization not available "
                "for this analysis.",
                styles["BodyText"]
            )
        )


    story.append(
        PageBreak()
    )


    story.append(
        Paragraph(
            "<b>AI Findings Summary</b>",
            styles["Heading2"]
        )
    )

    summary = (
        f"The uploaded image was analyzed using MedVisionAI. "
        f"The detected modality was <b>{detected_modality}</b> "
        f"with a confidence of <b>{modality_confidence:.2f}%</b>. "
        f"Following user verification, the final modality selected "
        f"was <b>{final_modality}</b>. "
        f"The disease classification model predicts "
        f"<b>{disease_result['prediction']}</b> "
        f"with a confidence of "
        f"<b>{disease_result['confidence']:.2f}%</b>."
    )

    story.append(
        Paragraph(
            summary,
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1, 20)
    )


    story.append(
        Paragraph(
            "<b>Disclaimer</b>",
            styles["Heading2"]
        )
    )

    disclaimer = (
        "This report is generated by an AI-powered medical imaging "
        "analysis system. The results are intended for educational, "
        "research, and decision-support purposes only. "
        "This report should not be used as a substitute for "
        "professional medical diagnosis, treatment, or clinical judgment."
    )

    story.append(
        Paragraph(
            disclaimer,
            styles["BodyText"]
        )
    )

    doc.build(story)

    return pdf_path