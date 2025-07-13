# Deployment Guide

## Streamlit Community Cloud
1. Push your code to GitHub.
2. Go to https://share.streamlit.io and follow the deployment wizard.
3. Set up secrets and config as needed.
4. Enjoy your deployed app!

## Heroku
1. Install Heroku CLI
2. Run: heroku create your-app-name
3. Run: git push heroku main
4. Run: heroku open

## Local Deployment
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run: streamlit run streamlit_app.py

## Configuration
- Use .env for secrets and environment variables.
- Use .streamlit/config.toml for Streamlit settings.

## Troubleshooting
- Check requirements.txt for missing dependencies.
- Ensure all files are present in the repo.
- For large files, use Git LFS or cloud storage.

## Support
- Open an issue on GitHub for help. 