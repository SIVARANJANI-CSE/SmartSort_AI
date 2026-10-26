# Image classification studio

A local Streamlit web app for training an image classifier from labeled folders and using it to classify new images. Folder names are the labels (the included dataset uses `O` and `R`). This project trains its own HOG-feature linear classifier from the supplied images; it does not require a remote or third-party model service.

The app uses HOG image features and a NumPy-trained linear classifier. It evaluates on `TEST` when present; otherwise it reserves 20% of the training images for validation. The default limit is 1,000 images per label per split, which can be changed in the app.

## Run

From this folder, install dependencies and launch the app:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in a browser. Select the included dataset or upload a ZIP with this layout:

```text
my_dataset.zip
  TRAIN/
    label_one/
      image1.jpg
    label_two/
      image2.jpg
  TEST/                 # optional
    label_one/
    label_two/
```

An extra top-level folder inside the ZIP is allowed. Supported image formats are JPG, JPEG, PNG, BMP, and WEBP. After training, the app shows the validation score and lets you classify images in the browser or download the trained model. Successful training results are also saved to `models/image_classifier.joblib`, then loaded automatically when the app starts again.

The included dataset uses `O` for organic waste and `R` for recyclable waste. The app displays a model score, not a calibrated probability, and waste-sorting rules vary by location. Validate predictions against representative images and check local guidance. Keep the saved model file private and only load model files you trust.

## Publish on GitHub and Streamlit Community Cloud

The repository ignores `.venv/` and `DATASET/` so large local dependencies and image collections are not uploaded. The saved model in `models/image_classifier.joblib` is included so a deployed app can classify images immediately. To train a different model after deployment, upload a ZIP containing labeled `TRAIN/<label>/<image>` folders.

Create an empty repository on GitHub. From this project folder, run these commands in the VS Code PowerShell terminal:

```powershell
git init
git add .gitignore app.py image_classifier.py requirements.txt README.md .streamlit/config.toml models/image_classifier.joblib
git status
git commit -m "Add SmartSort AI waste classifier"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with the GitHub account and repository name. Then open Streamlit Community Cloud, sign in with GitHub, choose **Create app**, select this repository and the `main` branch, set the app file to `app.py`, and deploy.
