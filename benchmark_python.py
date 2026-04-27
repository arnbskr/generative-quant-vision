import torch
import time

# Import des modèles
from autoencoder_model import GAF_Autoencoder, GAF_Classifier

def run_benchmark():
    print("[INIT] Chargement du modèle en Python...")
    
    # Chargement de l'architecture
    base_ae = GAF_Autoencoder()
    base_ae.load_state_dict(torch.load('Modeles/quant_gaf_autoencoder.pth', map_location='cpu'))
    
    model = GAF_Classifier(base_ae)
    model.eval() # Mode évaluation (désactive le dropout, très important pour la vitesse)

    # Tenseur factice 20x20 (Exactement comme dans main.cpp)
    dummy_input = torch.rand(1, 1, 20, 20)

    # "Chauffe" du modèle (Warm-up), en Python, la toute première exécution charge des bibliothèques en mémoire. On la fait tourner une fois "à vide" pour que la mesure soit juste.
    with torch.no_grad():
        _ = model(dummy_input)

    # Chronométrage
    start_time = time.perf_counter()
    
    with torch.no_grad(): # Désactive le calcul des gradients (accélère l'inférence)
        output = model(dummy_input)
        
    end_time = time.perf_counter()

    # Calculs et Affichage
    # perf_counter renvoie des secondes. On multiplie par 1 000 000 pour les microsecondes.
    duration_us = (end_time - start_time) * 1_000_000
    proba = output.item() * 100.0

    print("========================================")
    print(f"Prediction (Proba de hausse) : {proba:.4f} %")
    print(f"Latence d'inference (Python) : {duration_us:.0f} microsecondes")
    print("========================================")

if __name__ == "__main__":
    run_benchmark()