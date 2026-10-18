# Ciel visible — planificateur de nuit

    pip install -r requirements.txt
    python app.py        # ou : streamlit run app.py

- Premier lancement : télécharge le catalogue OpenNGC dans `data/` (internet requis une fois).
- Interface pensée pour le téléphone : onglets Plans / Réglages / Ciel / Objets / Partage.
- Thème clair/sombre : suit l'appareil de chaque visiteur (menu ⋮ → Settings).
- Rien n'est enregistré sur le serveur : contacts et clés WhatsApp restent dans la session du visiteur.

## Déploiement (Streamlit Community Cloud)
Contacts WhatsApp préenregistrés (optionnel) : *Settings → Secrets* de l'appli, par exemple

    [[contacts]]
    name = "Marie"
    phone = "+33612345678"
    apikey = "123456"

⚠️ Tout visiteur de l'appli pourrait alors envoyer des messages à ces contacts : si tu utilises les
secrets, restreins l'accès à l'appli (partage privé par e-mail dans les réglages de l'appli).

Catalogue OpenNGC : CC-BY-SA 4.0 (Mattia Verga).