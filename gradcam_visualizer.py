import torch
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from pyts.image import GramianAngularField
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from autoencoder_model import GAF_Autoencoder, GAF_Classifier

def generate_gradcam():
    # Chargement du modèle hybride
    print("Chargement du modèle")
    base_ae = GAF_Autoencoder()
    model = GAF_Classifier(base_ae)
    
    # On recharge l'encodeur entraîné depuis le dossier Modeles/
    base_ae.load_state_dict(torch.load('Modeles/quant_gaf_autoencoder.pth', map_location='cpu'))

    # On décongèle les poids de l'encodeur pour que Grad-CAM puisse calculer ses dérivées
    for param in model.feature_extractor.parameters():
        param.requires_grad = True

    model.eval()

    # Récupération d'une période de test (Ex: le krach du COVID en mars 2020)
    print("Récupération des données boursières")
    data = yf.download("^GSPC", start="2020-02-15", end="2020-03-25", progress=False)
    close_prices = data['Close'].values.flatten()[-20:] # Les 20 derniers jours

    # Transformation en GAF
    gaf = GramianAngularField(image_size=20, method='summation')
    X_gaf = gaf.fit_transform(np.array([close_prices]))
    tensor_x = torch.tensor(X_gaf, dtype=torch.float32).unsqueeze(1)

    # Configuration de Grad-CAM, on cible la dernière couche de convolution de l'encodeur
    target_layers = [model.feature_extractor[3]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Génération de la Heatmap, le modèle s'attend à un input (Batch, Channel, H, W)
    grayscale_cam = cam(input_tensor=tensor_x, targets=None)[0, :]

    # Préparation de l'image de fond (GAF normalisée entre 0 et 1 pour l'affichage)
    img_norm = (X_gaf[0] - X_gaf[0].min()) / (X_gaf[0].max() - X_gaf[0].min())
    # Conversion en RGB simulé (3 canaux) car Grad-CAM l'exige pour la superposition
    rgb_img = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)

    # Superposition de la Heatmap sur l'image GAF
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Affichage et sauvegarde
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(close_prices, color='blue', marker='o')
    axes[0].set_title("Prix S&P 500 (20 jours)")
    axes[0].grid(True)

    axes[1].imshow(X_gaf[0], cmap='rainbow')
    axes[1].set_title("Image GAF Originale")

    axes[2].imshow(visualization)
    axes[2].set_title("Grad-CAM : Zones d'attention de l'IA")

    plt.savefig("Images/gradcam_analysis.png")
    print("Analyse terminée, image sauvegardée sous 'Images/gradcam_analysis.png'.")

if __name__ == "__main__":
    generate_gradcam()