import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from pyts.image import GramianAngularField
from skimage.metrics import structural_similarity as ssim

# Importation du modèle GAF_Autoencoder et de la fonction de préparation
from autoencoder_model import GAF_Autoencoder, prepare_dataloaders

def get_financial_data(ticker="^GSPC", window_size=20):
    print("Téléchargement et transformation des données")
    data = yf.download(ticker, start="2014-01-01", end="2024-01-01", progress=False)
    close_prices = data['Close'].values.flatten()
    
    X = np.array([close_prices[i:i+window_size] for i in range(len(close_prices) - window_size)])
    gaf = GramianAngularField(image_size=window_size, method='summation')
    return gaf.fit_transform(X)

def train_model():
    # Hyperparamètres
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Préparation des données
    X_gaf = get_financial_data()
    train_loader, val_loader, test_loader = prepare_dataloaders(X_gaf, batch_size=BATCH_SIZE)

    # Initialisation du modèle, de la perte et de l'optimiseur
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEntraînement sur : {device}")
    
    model = GAF_Autoencoder().to(device)
    criterion = nn.MSELoss() # Perte MSE classique pour la reconstruction
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    print("\nDébut de l'entraînement")
    for epoch in range(EPOCHS):
        # Mode Entraînement
        model.train()
        running_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x) # On compare la sortie à l'entrée
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Mode Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_val, _ in val_loader:
                batch_val = batch_val.to(device)
                outputs = model(batch_val)
                loss = criterion(outputs, batch_val)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Sauvegarde des poids du modèle
    torch.save(model.state_dict(), 'Modeles/quant_gaf_autoencoder.pth')
    print("Modèle sauvegardé sous 'Modeles/quant_gaf_autoencoder.pth'")

    # Évaluation sur le jeu de test et génération des figures
    print("\nÉvaluation sur le Test Set et Génération des figures")
    
    # Courbes d'apprentissage
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Perte d\'entraînement (MSE)')
    plt.plot(val_losses, label='Perte de validation (MSE)')
    plt.title('Courbes d\'apprentissage de l\'auto-encodeur')
    plt.xlabel('Époques')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('Images/learning_curves.png')
    
    # Prise d'un batch de test pour visualiser la reconstruction
    model.eval()
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    
    with torch.no_grad():
        reconstructed = model(images)
    
    # Calcul du SSIM sur la première image du batch
    orig_img = images[0].cpu().squeeze().numpy()
    recon_img = reconstructed[0].cpu().squeeze().numpy()
    
    # Plage de données : les valeurs GAF vont de -1 à 1 (soit une plage de 2)
    data_range = orig_img.max() - orig_img.min()
    ssim_value = ssim(orig_img, recon_img, data_range=data_range)
    
    # Affichage Original vs Reconstruction
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1 = axes[0].imshow(orig_img, cmap='rainbow')
    axes[0].set_title("Vraie dynamique de marché (Original)")
    fig.colorbar(ax1, ax=axes[0])
    
    ax2 = axes[1].imshow(recon_img, cmap='rainbow')
    axes[1].set_title(f"Reconstruction par l'IA\n(SSIM: {ssim_value:.4f})")
    fig.colorbar(ax2, ax=axes[1])
    
    plt.savefig('Images/reconstruction_comparison.png')
    print(f"\nÉvaluation terminée. SSIM de l'échantillon : {ssim_value:.4f}")
    print("Graphiques 'learning_curves.png' et 'reconstruction_comparison.png' générés.")

if __name__ == "__main__":
    train_model()