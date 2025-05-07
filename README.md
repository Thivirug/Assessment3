# 🩺 MedAI 
## Assessment3: Wound Segmentation & Area Calculation

A user-friendly Streamlit app for automatic wound segmentation and area measurement using deep learning and computer vision.

---

## 🚀 Features

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
├── App/                # Main application code
│   ├── Home.py         # Streamlit app entry point
│   ├── pages/          # Streamlit multipage support
│   ├── CalcArea.py     # Area calculation script
│   ├── Unet.py         # U-Net model definition
│   └── Assets/         # Static assets
│
├── Checkpoints/        # Trained model weights (Not included in repo)
├── data/               # Image datasets (Not included in repo)
├── requirements.txt    # Python dependencies
├── main.ipynb          # Comprehensive Jupyter notebook for data exploration, model training, and evaluation
├── LICENSE             # Project license
└── README.md           # Project documentation
```

---

## ⚙️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Download model weights:**
   The app will automatically download the model from Google Drive if not present.
3. **Run the app:**
   ```bash
   streamlit run App/Home.py
   ```

---

## 📝 Usage Guide

1. **Upload** a wound image with a visible 5x5 chessboard reference grid.
2. Click **"Generate Mask"** to segment the wound.
3. Click **"Show Mask on Image"** to visualise the mask outline.
4. Click **"Calculate Area"** to compute the wound area in cm².

> **Tip:** For best results, ensure good lighting and minimal occlusion of the reference grid.

---

## ❗ Notes

- The area calculation requires a clear 5x5 chessboard pattern in the image for scale. If not found, an error will be shown.
- All processing is local; your images are not uploaded to any server.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

