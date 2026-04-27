import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Préparation des données (DataLoader)

def prepare_dataloaders(X_gaf, batch_size=32):
    """
    Transforme les matrices Numpy en Tenseurs PyTorch et crée les splits.
    """
    # PyTorch attend des images sous la forme (Batch, Canaux, Hauteur, Largeur), donc on ajoute la dimension des canaux (1 canal car c'est en niveaux de gris/valeurs brutes).
    X_gaf = np.expand_dims(X_gaf, axis=1)
    
    # Conversion en tenseurs PyTorch (Float 32 bits)
    tensor_x = torch.tensor(X_gaf, dtype=torch.float32)
    
    # Division en Train (70%), Validation (15%), Test (15%)
    x_train, x_temp = train_test_split(tensor_x, test_size=0.3, random_state=42)
    x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)
    
    # Pour un auto-encodeur, la "cible" (Y) est la même que l'entrée (X)
    train_dataset = TensorDataset(x_train, x_train)
    val_dataset = TensorDataset(x_val, x_val)
    test_dataset = TensorDataset(x_test, x_test)
    
    # Création des DataLoaders pour l'entraînement par lots (batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders créés : {len(x_train)} images train, {len(x_val)} val, {len(x_test)} test.")
    return train_loader, val_loader, test_loader

# Architecture de l'auto-encodeur

class GAF_Autoencoder(nn.Module):
    def __init__(self):
        super(GAF_Autoencoder, self).__init__()
        
        # Encodeur : Compression de l'image
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Espace latent (Bottleneck) : L'image de 20x20 a été compressée en une représentation dense de 5x5 avec 32 filtres.
        
        # Décodeur : Reconstruction de l'image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x) # Passage dans l'encodeur
        decoded = self.decoder(encoded) # Passage dans le décodeur
        return decoded

# Petit test rapide pour vérifier que les dimensions sont correctes
if __name__ == "__main__":
    model = GAF_Autoencoder()
    print(model)
    # On simule un batch de 32 images de taille 1x20x20
    test_tensor = torch.randn(32, 1, 20, 20) 
    output = model(test_tensor)
    print(f"\nForme du tenseur d'entrée : {test_tensor.shape}")
    print(f"Forme du tenseur de sortie : {output.shape}") 
    # La sortie doit être strictement identique à l'entrée (32, 1, 20, 20)