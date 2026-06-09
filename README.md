# 🩺 MedVisionAI - A Multimodal Medical Imaging Analysis Platform 

**MedVisionAI** is a deep learning–powered medical imaging platform that automatically detects the type of scan uploaded, classifies the relevant disease, and generates an explainable, downloadable diagnostic report — all within a single interface.<br>
🎯**Motivation** - Radiological diagnosis is time-intensive and expertise-dependent. MedVisionAI was built to explore how AI can assist — not replace clinicians by providing fast, interpretable predictions across multiple imaging modalities from one unified tool.

<br>

## ✨ Features

| Feature | Description |
|---|---|
| Auto Modality Detection | Identifies scan type (Brain MRI / CT, Chest X-Ray / CT) automatically before any disease analysis |
| Human-in-the-Loop | User verifies or corrects the detected modality — the model proposes, the user decides |
| Multi-Disease Classification | Four fine-tuned models independently handle tumor, hemorrhage, pneumonia, and COVID-19 detection |
| Grad-CAM Explainability | Heatmap overlays embedded in the PDF report highlight regions that drove each prediction |
| PDF Report Generation | Downloadable diagnostic report with scan preview, Grad-CAM, modality info, class probabilities, and AI findings |

<br>

## 📸 Application Demo
<details>
<summary>🎬 Click to view MedVisionAI Demo</summary>

<br>

<img src="assets/demo.gif" width="100%" align="left">

</details>

<br>

## ⬇️ Project Workflow
1. **Upload** a medical scan — `.jpg`, `.jpeg`, `.png`, `.webp`, or `.dcm`
2. **Review** the auto-detected modality of the uploaded scan
3. **Confirm** if correct, or select the right modality from the dropdown
4. **Run Disease Analysis** to trigger the appropriate classifier
5. **View results** — prediction, confidence bar, and contextual alert
6. **Download** the full  PDF diagnostic report.

<br>


## 📚 Datasets used and Visualization
 
| Modality | Condition | Dataset |
|---|---|---|
| Chest X-Ray | Pneumonia | [Chest X-Ray Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Brain MRI | Tumor | [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Chest CT | COVID-19 | [SARS-CoV-2 CT Scan Dataset — Kaggle](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) |
| Brain CT | Hemorrhage | [Brain CT Hemorrhage Dataset — Kaggle](https://www.kaggle.com/datasets/abdulkader90/brain-ct-hemorrhage-dataset) |

<table>
<tr>
<td align="center">
<img src="assets\Chest Xray\class distribution.png" width="450"><br>
</td>

<td align="center">
<img src="assets\Brain MRI\class distribution.png" width="450"><br>

</td>
</tr>

<tr>
<td align="center">
<img src="assets\Chest CT\class distribution.png" width="450"><br>
</td>

<td align="center">
<img src="assets\Brain CT\class distribution.png" width="450"><br>
</td>
</tr>
</table>

<br>

## 🤖 Supported Models 
 
| Modality | Model | Disease Detected | Output Classes |
|---|---|---|---|
| Brain MRI | DenseNet-121 | Brain Tumor | Glioma, Meningioma, No Tumor, Pituitary |
| Brain CT | EfficientNet-B0 | Intracranial Hemorrhage | Normal, Hemorrhage |
| Chest X-Ray | ResNet-50 | Pneumonia | Normal, Pneumonia |
| Chest CT | ConvNeXt-Tiny | COVID-19 | COVID, Non-COVID |

<br>


## 📈 Model Performance

| Model | Accuracy |
|---------|---------|
| Brain MRI Tumor Detection (DenseNet121) | 93.75% |
| Brain CT Hemorrhage Detection (EfficientNet-B0) | 98.46% |
| Chest X-Ray Pneumonia Detection (ResNet50) | 97.89% |
| Chest CT COVID Detection (ConvNeXt-Tiny) | 96.98% |
| Modality Classifier (EfficientNet-B0) | 99.9% |

<table>
<tr>
<td align="center">
<img src="assets\Brain MRI\train loss curve.png" width="450"><br></td>

<td align="center">
<img src="assets\Brain CT\training loss curve.png" width="450"><br>
</td>


</tr>
<tr>
</td>
<td align="center">
<img src="assets\Chest Xray\Training loss curve.png" width="450"><br>
</td>
<td align="center">
<img src="assets\Chest CT\training loss curve.png" width="450"><br>
</td>
</tr>

</table>

<br>

## 🚀 Explainable AI — Grad-CAM

**Why it matters:** Raw predictions alone are not sufficient in a clinical context. Grad-CAM makes the model's reasoning visible by highlighting the anatomical regions — a lesion, an opacification, a bleed — that most influenced the classification. These heatmaps are embedded directly in the downloadable PDF report for radiologist review.

<p align="left">
  <img src="assets/Brain CT/grad-cam hemorrhage.png" width="900">
</p>

<br>

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python|
| Deep Learning | PyTorch, TorchVision |
| Image Processing | OpenCV, Pillow |
| Report Generation | ReportLab |
| Frontend | Streamlit |
| Training Environment | Google Colab |

<br>

## 📂 Project Structure

```
MedVisionAI/
│
├── app.py                         # Main Streamlit application
│
├── utils/
│   ├── modality_detection.py      # EfficientNet-B0 modality classifier
│   ├── router.py                  # Routes input to correct disease model
│   ├── brain_mri.py               # DenseNet-121 · Tumor (4-class)
│   ├── brain_ct.py                # EfficientNet-B0 · Hemorrhage
│   ├── chest_xray.py              # ResNet-50 · Pneumonia
│   ├── chest_ct.py                # ConvNeXt-Tiny · COVID-19
│   ├── gradcam.py                 # Grad-CAM visualization engine
│   └── report_generator.py        # PDF report builder
│
├── notebooks/                     # Training notebooks (Google Colab)
│   ├── brain_tumor_detection.ipynb
│   ├── hemorrhage_detection.ipynb
│   ├── pneumonia_detection.ipynb
│   ├── covid_detection.ipynb
│   └── modality_detection.ipynb
│
├── assets/                        # Images and GIFs for README
│
└── requirements.txt
```

<br>

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/agrawalanshika/MedVisionAI.git
cd MedVisionAI
```
 
**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        
```
 
**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add model weights**
 
Place your trained `.pth` files inside the `models/` directory:
```
models/modality_classifier.pth
models/brain_mri_densenet121.pth
models/brain_ct_hemorrhage.pth
models/resnet50_pneumonia_v2.pth
models/chest_ct_convnext_tiny.pth
```
 
**5. Run the app**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

<br>

## 🔮 Future Work
 
- **Lesion segmentation** — Move beyond classification to pixel-level detection using U-Net or SAM
- **Uncertainty quantification** — Add Monte Carlo Dropout for calibrated confidence intervals
- **Expanded modality support** — Abdominal CT, spine MRI, retinal fundus imaging
- **DICOM metadata parsing** — Display scanner info, slice thickness, and patient metadata from `.dcm` files
- **Cloud deployment** —  Using Streamlit Community Cloud, Hugging Face Spaces, or AWS

<br>

## ⚠️ Disclaimer
 
MedVisionAI is developed strictly for **educational and research purposes**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis, treatment, or medical judgment. All outputs should be reviewed by a qualified radiologist or physician. The authors assume no liability for clinical decisions made based on this system.

<br>

## 👩‍💻 Author
 
## Anshika Agrawal
