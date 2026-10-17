# Ciel visible — planificateur de nuit

    pip install -r requirements.txt
    python app.py        # ou : streamlit run app.py

- Premier lancement : télécharge le catalogue OpenNGC dans `data/` (internet requis une fois).
- `config.json` (créé automatiquement) : mode sombre, contacts WhatsApp, clés API (en clair).
- Envoi direct WhatsApp : CallMeBot (chaque destinataire active le bot depuis son WhatsApp
  et te donne sa clé) ou WhatsApp Business Cloud API (Meta).
- Catalogue OpenNGC : CC-BY-SA 4.0 (Mattia Verga).