import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from train import get_financial_data
from autoencoder_model import prepare_dataloaders
from vae_model import GAF_VAE

# Fonction de perte VAE
def vae_loss_function(recon_x, x, mu, logvar):
    # Erreur de reconstruction (MSE)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # Divergence KL (force l'espace latent à former une loi normale)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_and_generate():
    print("Préparation des données")
    X_gaf = get_financial_data()
    train_loader, _, _ = prepare_dataloaders(X_gaf, batch_size=32)

    device = torch.device("cpu")
    model = GAF_VAE(latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nEntraînement du Variational Autoencoder")
    EPOCHS = 30
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_x)
            loss = vae_loss_function(recon_batch, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss Totale: {train_loss / len(train_loader.dataset):.4f}")

    # Sauvegarde du modèle
    os.makedirs('Modeles', exist_ok=True)
    torch.save(model.state_dict(), 'Modeles/quant_vae.pth')
    print("Modèle sauvegardé sous 'Modeles/quant_vae.pth'")

    print("\nGénération de données synthétiques")
    model.eval()
    
    # On tire 3 vecteurs latents au hasard dans une gaussienne
    random_latent_vectors = torch.randn(3, 32).to(device)
    
    with torch.no_grad():
        synthetic_gaf_images = model.decode(random_latent_vectors)
    
    os.makedirs('Images', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Création de marchés synthétiques inédits par le VAE", fontsize=16)
    
    for i in range(3):
        img = synthetic_gaf_images[i].squeeze().numpy()
        axes[i].imshow(img, cmap='rainbow')
        axes[i].set_title(f"Univers Parallèle n°{i+1}")
        axes[i].axis('off')
    
    plt.savefig("Images/synthetic_markets_vae.png")
    print("Image sauvegardée sous 'Images/synthetic_markets_vae.png'.")

if __name__ == "__main__":
    train_and_generate()