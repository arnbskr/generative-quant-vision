import streamlit as st
import torch
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from pyts.image import GramianAngularField
from skimage.metrics import structural_similarity as ssim
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from autoencoder_model import GAF_Autoencoder, GAF_Classifier
from vae_model import GAF_VAE

# Configuration de la page
st.set_page_config(page_title="Quant Vision AI", layout="wide")
st.title("Generative Quant Vision System")
st.markdown("*Plateforme d'analyse prédictive et de stress-testing par vision artificielle : du signal brut au déploiement basse latence en C++.*")
st.markdown("---")

# Initialisation de la session pour stocker les données et les états de l'application
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True,
        'prices': None, 'gaf': None, 'tensor': None,
        'show_gaf': False, 'inference_done': False, 
        'proba': None, 'ssim': None, 'recon_np': None,
        'gradcam_img': None, 'vae_samples': None
    })

# Chargement des modèles (Auto-encodeur, Classifieur, VAE)
@st.cache_resource
def load_models():
    # Auto-encodeur
    ae = GAF_Autoencoder()
    ae.load_state_dict(torch.load('Modeles/quant_gaf_autoencoder.pth', map_location='cpu'))
    
    # Classifieur (Inférence et Grad-CAM)
    clf = GAF_Classifier(ae)
    # Décongélation des poids pour permettre le calcul des gradients par Grad-CAM
    for param in clf.feature_extractor.parameters():
        param.requires_grad = True
    clf.eval()
    
    # VAE (Génération)
    vae = GAF_VAE()
    vae.load_state_dict(torch.load('Modeles/quant_vae.pth', map_location='cpu'))
    vae.eval()
    
    return ae, clf, vae

ae_model, clf_model, vae_model = load_models()

# Ingestion des données financières et transformation en GAF
st.sidebar.header("🕹️ Ingestion des Données")
ticker = st.sidebar.text_input("Symbole (Ex: ^GSPC, AAPL, TSLA)", "^GSPC")
# Blocage à la date du jour pour éviter les erreurs de données futures
target_date = st.sidebar.date_input(
    "Date cible (Fin de fenêtre)", 
    value=datetime.date.today(), 
    max_value=datetime.date.today()
)

if st.sidebar.button("📥 Charger les données"):
    with st.spinner("Téléchargement des cours..."):
        data = yf.download(ticker, end=target_date, period="60d", progress=False)
        if not data.empty:
            # Extraction des 20 derniers jours de clôture
            prices = data['Close'].values.flatten()[-20:]
            gaf_obj = GramianAngularField(image_size=20, method='summation')
            img_gaf = gaf_obj.fit_transform(np.array([prices]))
            
            # Stockage dans la session
            st.session_state.prices = prices
            st.session_state.gaf = img_gaf[0]
            st.session_state.tensor = torch.tensor(img_gaf).unsqueeze(1).float()
            
            # Réinitialisation des états visuels
            st.session_state.show_gaf = False
            st.session_state.inference_done = False
            st.session_state.gradcam_img = None
            st.session_state.vae_samples = None
            st.sidebar.success(f"Données de {ticker} prêtes !")

# Navigation par onglets
tab1, tab2, tab3, tab4 = st.tabs(["📈 Vision & Inférence", "🔍 Interprétabilité", "🎨 Génération VAE", "⚡ Stats C++"])

if st.session_state.prices is not None:
    
    # Onglet 1 : Affichage de la transformation GAF et inférence
    with tab1:
        st.subheader("1. Transformation Spatiale (GAF)")
        if st.button("👁️ Afficher la transformation 1D -> 2D", key="btn_show_gaf"):
            st.session_state.show_gaf = True
            
        if st.session_state.show_gaf:
            colA, colB = st.columns(2)
            with colA:
                fig1, ax1 = plt.subplots(figsize=(6,4))
                ax1.plot(st.session_state.prices, marker='o', color='#1f77b4', linewidth=2)
                ax1.set_title("1D : Série temporelle (Prix)")
                ax1.grid(alpha=0.3)
                st.pyplot(fig1)
            with colB:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                im = ax2.imshow(st.session_state.gaf, cmap='rainbow')
                ax2.set_title("2D : Image GAF (Corrélations)")
                plt.colorbar(im, ax=ax2)
                st.pyplot(fig2)

            st.markdown("---")
            st.subheader("2. Intelligence Artificielle")
            if st.button("🚀 Lancer l'inférence (Prédiction J+1)", key="btn_run_inf"):
                with st.spinner("Analyse du pattern..."):
                    with torch.no_grad():
                        recon = ae_model(st.session_state.tensor)
                        st.session_state.proba = clf_model(st.session_state.tensor).item()
                        st.session_state.recon_np = recon[0].squeeze().numpy()
                        orig = st.session_state.gaf
                        # Calcul du SSIM pour détecter les anomalies structurelles
                        st.session_state.ssim = ssim(orig, st.session_state.recon_np, data_range=orig.max()-orig.min())
                        st.session_state.inference_done = True
            
            if st.session_state.inference_done:
                c1, c2, c3 = st.columns(3)
                c1.metric("Direction prédite", "📈 HAUSSE" if st.session_state.proba > 0.5 else "📉 BAISSE")
                c2.metric("Confiance", f"{st.session_state.proba*100:.2f}%")
                c3.metric("Indice de Normalité (SSIM)", f"{st.session_state.ssim:.4f}")
                
                if st.session_state.ssim < 0.82:
                    st.warning("⚠️ Structure atypique détectée : Risque d'anomalie élevé.")
                else:
                    st.success("✅ Structure standard : Régime de marché connu.")

    # Onglet 2 : Interprétabilité avec Grad-CAM
    with tab2:
        st.header("Analyse des zones d'attention")
        st.write("Quels motifs de la matrice ont provoqué la décision de l'IA ?")
        
        if st.button("🎯 Calculer le Grad-CAM", key="btn_gradcam"):
            with st.spinner("Calcul des gradients (Haute Résolution)..."):
                # Cible la première couche pour éviter l'effet flou du pooling
                target_layers = [clf_model.feature_extractor[0]]
                cam = GradCAM(model=clf_model, target_layers=target_layers)
                
                input_tensor = st.session_state.tensor
                input_tensor.requires_grad = True
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                
                # Normalisation pour l'affichage
                img_norm = (st.session_state.gaf - st.session_state.gaf.min()) / (st.session_state.gaf.max() - st.session_state.gaf.min())
                rgb_img = np.repeat(img_norm[:, :, np.newaxis], 3, axis=2)
                st.session_state.gradcam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
        if st.session_state.gradcam_img is not None:
            st.image(st.session_state.gradcam_img, caption="Carte d'attention Grad-CAM", width=600)

    # Onglet 3 : Génération de scénarios alternatifs avec le VAE
    with tab3:
        st.header("Simulation de Scénarios Alternatifs")
        st.write("Génère des variantes réalistes basées sur la signature latente de la période actuelle.")
        
        if st.button("🎲 Générer 3 Variantes", key="btn_vae_gen"):
            with torch.no_grad():
                mu, logvar = vae_model.encode(st.session_state.tensor)
                samples = [vae_model.decode(vae_model.reparameterize(mu, logvar)).squeeze().numpy() for _ in range(3)]
                st.session_state.vae_samples = samples
                
        if st.session_state.vae_samples is not None:
            cols = st.columns(3)
            for i, s in enumerate(st.session_state.vae_samples):
                with cols[i]:
                    # Affichage 2D en couleur
                    fig_2d, ax_2d = plt.subplots(figsize=(4,4))
                    ax_2d.imshow(s, cmap='rainbow')
                    ax_2d.set_title(f"Scénario {i+1} (2D)")
                    ax_2d.axis('off')
                    st.pyplot(fig_2d)
                    
                    # Extraction et affichage de la courbe 1D (diagonale)
                    diag_1d = np.diagonal(s)
                    fig_1d, ax_1d = plt.subplots(figsize=(4,2))
                    ax_1d.plot(diag_1d, color='#9b59b6', linewidth=2)
                    ax_1d.set_title("Prix reconstruit (1D)")
                    ax_1d.grid(alpha=0.2)
                    st.pyplot(fig_1d)

else:
    st.info("👈 Commencez par choisir un symbole et cliquez sur 'Charger les données' dans la barre latérale.")

# === ONGLET 4 : C++ ===
with tab4:
    st.header("Déploiement Industriel : Moteur C++ (LibTorch)")
    st.write("En finance quantitative, les modèles doivent réagir en quelques millisecondes. L'architecture Python a donc été exportée en `TorchScript` pour être exécutée de manière autonome par un moteur C++ basse latence.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.success("✅ Modèle exporté et validé en production")
        st.metric("Latence d'inférence mesurée", "51.38 ms")
        st.metric("Temps brut", "51 377 microsecondes")
        
    with col2:
        st.write("💻 **Sortie standard du programme C++ (`./quant_engine`) :**")
        st.code("""
(venv) arb@fedora:~/.../generative-quant-vision/build$ ./quant_engine 
[SUCCESS] Modele Quantitatif charge en C++
========================================
Prediction (Proba de hausse) : 54.8889 %
Latence d'inference          : 51377 microsecondes
========================================
        """, language="bash")