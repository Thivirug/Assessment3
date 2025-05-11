# 🩺 MedAI 
## Wound Segmentation & Area Calculation

A user-friendly Streamlit app for automatic wound segmentation and area measurement using deep learning and computer vision.

---

## ✨ Features

- 📤 **Upload wound images** with a 5x5 chessboard reference grid
- 🤖 **Automatic segmentation** using a trained U-Net model
- 🖼️ **Visualise masks** and outlines on the original image
- 📏 **Calculate wound area** in cm² using the reference grid for scale
- ⚠️ **Robust error handling** and clear user feedback

---

## 🗂️ Project Structure

```
Assessment3/
│
├── App/                      # Main application code
│   ├── .streamlit            # secrets configuration directory (not included in repo)
│        ├── secrets.toml     # Streamlit configuration file
│   ├── Home.py               # Streamlit app entry point
│   ├── pages/                # Streamlit multipage support
│        ├── 1_About.py       # About page
│        ├── 2_Contact.py     # Contact page
│        ├── 3_Diagnose.py    # Diagnosis page
│   ├── CalcArea.py           # Area calculation script
│   ├── CalcAreaLogic.txt     # Area calculation logic
│   ├── Unet.py               # U-Net model definition
│   ├── IoU.py                # Custom IoU metric
│   ├── Loss.py               # Custom loss function
│   └── Assets/               # Static assets
│
├── Checkpoints/              # Trained model (Not included in repo)
├── data/                     # Image datasets (Not included in repo)
├── requirements.txt          # Python dependencies
├── packages.txt              # linux packages for opencv
├── final notebook 1.ipynb    # Comprehensive Jupyter notebook for data exploration, model training, and evaluation -> MODEL1
├── final notebook 2.ipynb    # Comprehensive Jupyter notebook for data exploration, model training, and evaluation -> MODEL2
├── .gitignore                # Git ignore file
├── LICENSE                   # Project license
└── README.md                 # Project documentation
```

---

## ⚙️ Streamlit App

This is the link to the deployed Streamlit app: [MedAI Wound Segmentation](https://assessment3-woundsegv2.streamlit.app/)

---

## 📝 Usage Guide

1. **Upload** a wound image with a visible 5x5 chessboard reference grid.
2. Click **"Generate Mask"** to segment the wound.
3. Click **"Show Mask on Image"** to visualise the mask outline.
4. Click **"Calculate Area"** to compute the wound area in cm².

> **Tip:** For best results, ensure good lighting, use a black and white reference grid, and maintain a considerable gap between the wound and the reference.

---

## ❗ Notes

- The area calculation requires a clear 5x5 chessboard pattern in the image for scale. If not found, an error will be shown.
- All processing is local; your images are not uploaded to any server.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

