import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import os
from pyts.image import GramianAngularField
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from autoencoder_model import GAF_Autoencoder, GAF_Classifier

def get_labeled_financial_data(ticker="^GSPC", window_size=20):
    data = yf.download(ticker, start="2014-01-01", end="2024-01-01", progress=False)
    close_prices = data['Close'].values.flatten()
    
    X, y = [], []
    for i in range(len(close_prices) - window_size - 1):
        # Fenêtre de 20 jours
        X.append(close_prices[i:i+window_size])
        
        # Label : le prix du 21ème jour est-il supérieur au 20ème ?
        if close_prices[i+window_size] > close_prices[i+window_size-1]:
            y.append(1.0) # Hausse
        else:
            y.append(0.0) # Baisse
            
    X = np.array(X)
    y = np.array(y)
    
    gaf = GramianAngularField(image_size=window_size, method='summation')
    X_gaf = gaf.fit_transform(X)
    
    return X_gaf, y

def train_predictor():
    BATCH_SIZE = 32
    EPOCHS = 30
    
    print("Préparation des données étiquetées")
    X_gaf, y = get_labeled_financial_data()
    
    X_tensor = torch.tensor(X_gaf, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    
    print("Chargement de l'auto-encodeur pré-entraîné")
    base_ae = GAF_Autoencoder()
    base_ae.load_state_dict(torch.load('Modeles/quant_gaf_autoencoder.pth', map_location='cpu'))
    
    print("Création et entraînement du Classifieur")
    model = GAF_Classifier(base_ae)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")
            
    print("Exportation du modèle vers TorchScript (pour C++)")
    model.eval()
    example_input = torch.rand(1, 1, 20, 20)
    traced_script_module = torch.jit.trace(model, example_input)
    
    # On sauvegarde le modèle dans le dossier Modeles/
    os.makedirs('Modeles', exist_ok=True)
    traced_script_module.save("Modeles/quant_classifier_cpp.pt")
    print("Modèle exporté sous 'Modeles/quant_classifier_cpp.pt', prêt pour le moteur d'inférence.")

if __name__ == "__main__":
    train_predictor()