@echo off
cd /d "%~dp0"

echo ========================================================
echo      XENOPUS MORPHOMETRIC PIPELINE (M2 PROJECT)
echo ========================================================
echo.
echo Lancement de l'interface d'analyse en cours...
echo Veuillez patienter, une page web va s'ouvrir.
echo.

streamlit run src/app.py

if %errorlevel% neq 0 (
    echo.
    echo UNE ERREUR S'EST PRODUITE !
    pause
)