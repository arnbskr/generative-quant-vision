import torch
import torch.nn as nn

class GAF_VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(GAF_VAE, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
        
        # Au lieu de sortir un vecteur fixe, on sort une moyenne et un log-variance
        self.fc_mu = nn.Linear(32 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(32 * 5 * 5, latent_dim)
        
        # Décodeur
        self.fc_decode = nn.Linear(latent_dim, 32 * 5 * 5)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Tanh() # Les valeurs GAF oscillent entre -1 et 1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)   
        return mu + eps * std         

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 32, 5, 5) 
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar