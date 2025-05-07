# Assessment3: Wound Segmentation and Area Calculation

This project provides a Streamlit web application for automatic wound segmentation and area calculation using deep learning (U-Net) and computer vision techniques.

## Features
- Upload wound images with a reference grid (5x5 chessboard pattern)
- Generate segmentation masks using a trained U-Net model
- Visualise the mask and its outline on the original image
- Calculate the wound area in cm² using the reference grid for scale
- User-friendly interface with error handling

## Project Structure
- `App/` - Main application code
  - `pages/` - Streamlit multipage support
  - `CalcArea.py` - Area calculation logic
  - `Unet.py` - U-Net model definition
  - `Assets/` - Static assets (if any)
- `Checkpoints/` - Trained model weights (`unet_best_model.keras`)
- `data/` - Image datasets
- `requirements.txt` - Python dependencies
- `main.ipynb` - Example notebook for predictions

## Setup
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

## Usage
- Upload a wound image with a visible 5x5 chessboard reference grid.
- Click "Generate Mask" to segment the wound.
- Click "Show Mask on Image" to visualize the mask outline.
- Click "Calculate Area" to compute the wound area in cm².

## Notes
- The area calculation requires a clear 5x5 chessboard pattern in the image for scale. If not found, an error will be shown.
- For best results, ensure good lighting and minimal occlusion of the reference grid.

## License
This project is for academic use only.
