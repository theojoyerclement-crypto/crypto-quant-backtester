# Commandes Git (Windows cmd)

À exécuter depuis la racine du dépôt :

    git status
    git add setup_files.py backtester_fixed.py README.md COMMANDES_GIT.md
    git commit -m "Ajoute le backtester crypto sans fuite de donnees"
    git push origin main

Pour vérifier le commit avant l'envoi :

    git log -1 --oneline
    git status

Ces commandes supposent que le dépôt distant `origin` et l'authentification Git sont déjà configurés.
