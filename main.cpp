#include <torch/script.h>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../Modeles/quant_classifier_cpp.pt");
        std::cout << "[SUCCESS] Modele Quantitatif charge en C++\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Erreur de chargement du modele. Verifiez le chemin.\n";
        return -1;
    }

    // Création d'un tenseur factice simulant une image GAF (1 batch, 1 canal, 20x20)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 1, 20, 20}));

    // On fait tourner le modèle une fois "à vide" pour compiler le graphe JIT
    module.forward(inputs);

    // Mesure de la vraie latence d'inference (Crucial en HFT)
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execution du modele (Vraie passe)
    at::Tensor output = module.forward(inputs).toTensor();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Affichage des resultats
    float probabilite_hausse = output.item<float>();
    std::cout << "========================================\n";
    std::cout << "Prediction (Proba de hausse) : " << probabilite_hausse * 100.0 << " %\n";
    std::cout << "Latence d'inference          : " << duration.count() << " microsecondes\n";
    std::cout << "========================================\n";

    return 0;
}