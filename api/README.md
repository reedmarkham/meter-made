# FastAPI Parking Ticket Prediction App

This FastAPI app serves a machine learning model (`model.pkl`) for predicting expired meter parking tickets in Washington, DC. The model is trained in a Colab notebook and uploaded manually to Google Storage.

## Deployment

The app is deployed to **Cloud Run** using **Cloud Build** with a trigger connected to this repository. Changes pushed to the repository automatically trigger a build and deployment to Cloud Run.

### Model Training and Upload

The model is trained using the notebook available [here](https://nbviewer.org/github/reedmarkham/meter-made/blob/main/meter_made.ipynb). After training, the `model.pkl` file is manually uploaded to Google Storage.

### Serving the Model

Upon deployment, the FastAPI app loads the model from Google Storage and exposes an endpoint for making predictions. You can interact with the model through this API.

For more details on how the app and model work, refer to the notebook and the FastAPI documentation.