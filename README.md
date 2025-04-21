# Darija Classifier

Un système de classification d'intentions pour les textes en Darija (arabe marocain) utilisant le modèle ATLAS2B. Cette application analyse les messages en Darija et détermine l'intention de l'utilisateur, ce qui est particulièrement utile pour les centres d'appels et les systèmes de service client automatisés.

## 🔗 Liens

- [HuggingFace Space](https://huggingface.co/spaces/mohamedGOUALI/TEMPORAIRE) - **⚠️ Démo en cours de développement (non fonctionnelle)**
- [Modèle ATLAS2B](https://huggingface.co/mohamedGOUALI/ATLAS2B_test) - Modèle de langage utilisé

##  Statut du projet

**Ce projet est actuellement en développement actif et la démo n'est pas encore fonctionnelle.**
##  Fonctionnalités prévues

- Interface utilisateur intuitive pour la saisie de texte en Darija
- Classification automatique des intentions utilisateur
- Historique des conversations
- Analyse des messages avec affichage des résultats de classification
- Fonctionnement asynchrone pour améliorer les performances

## 🧠 Catégories d'intentions

Le système est conçu pour classifier les messages dans des catégories prédéfinies par le UseCase.
## 🛠️ Technologies utilisées

- **Backend** : FastAPI, Python 3.9+
- **IA** : Hugging Face Transformers, ATLAS2B model
- **Frontend** : HTML, CSS, JavaScript, Bootstrap
- **Hébergement** : HuggingFace Spaces
