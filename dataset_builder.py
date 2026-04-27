import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

# Téléchargement des données (S&P 500)
print("Téléchargement des données")
ticker = "^GSPC" # Symbole du S&P 500
data = yf.download(ticker, start="2014-01-01", end="2024-01-01")

# Nous prenons uniquement le prix de clôture pour simplifier l'auto-encodeur
close_prices = data['Close'].values.flatten()

# Création des fenêtres glissantes
window_size = 20 # 20 jours = image 20x20 pixels
X = []

print(f"Création des fenêtres glissantes de taille {window_size}")
for i in range(len(close_prices) - window_size):
    X.append(close_prices[i:i+window_size])
X = np.array(X)

# Transformation mathématique en images (GAF)
print("Transformation des séries en images GAF")
gaf = GramianAngularField(image_size=window_size, method='summation')
X_gaf = gaf.fit_transform(X)

print(f"Dimension finale du dataset : {X_gaf.shape}") 
# Devrait afficher (N, 20, 20) où N est le nombre d'images générées

# Visualisation pour vérifier le résultat
plt.figure(figsize=(6, 6))
plt.imshow(X_gaf[0], cmap='rainbow') 
plt.title("Image GAF - 20 premiers jours du S&P 500")
plt.colorbar()
plt.savefig("Images/sample_gaf.png")
plt.show()