# Callcenter-in-darija

Un système de classification d'intentions pour les textes en Darija (arabe marocain) utilisant le modèle AtlasChat 2B. Cette application analyse les messages en Darija et détermine l'intention de l'utilisateur, ce qui est particulièrement utile pour les centres d'appels et les systèmes de service client automatisés.

## 🔗 Liens

- [HuggingFace Space](https://huggingface.co/spaces/mohamedGOUALI/TEMPORAIRE) - **⚠️ Démo en cours de développement (non fonctionnelle)**
- [Modèle ATLAS2B](https://huggingface.co/mohamedGOUALI/ATLAS2B_test) - Modèle de langage utilisé

## 🚧 Statut du projet

**Ce projet est actuellement en développement actif et la démo n'est pas encore fonctionnelle.**

Problèmes connus :
- L'interface utilisateur ne s'affiche pas correctement dans l'environnement HuggingFace Spaces
- Des ajustements sont nécessaires pour assurer la compatibilité avec l'environnement d'hébergement
- La connexion avec le modèle ATLAS2B est établie mais l'interface utilisateur rencontre des problèmes d'affichage

## ✨ Fonctionnalités prévues

- Interface utilisateur intuitive pour la saisie de texte en Darija
- Classification automatique des intentions utilisateur
- Historique des conversations
- Analyse des messages avec affichage des résultats de classification
- Fonctionnement asynchrone pour améliorer les performances

## 🧠 Catégories d'intentions

Le système est conçu pour classifier les messages dans des catégories prédéfinies par le UseCase(e.g, problème de facturation , ...)

## 🛠️ Technologies utilisées

- **Backend** : FastAPI, Python 3.9+
- **IA** : Hugging Face Transformers, Atlas-chat by MBZUAI
- **Frontend** : HTML, CSS, JavaScript, Bootstrap
- **Hébergement** : HuggingFace Spaces

