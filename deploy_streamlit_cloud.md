# 🚀 Deploy to Streamlit Cloud - Step by Step

## Prerequisites
- GitHub account
- Your code ready to deploy

## Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create GitHub Repository**:
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name it (e.g., "neural-data-analysis")
   - Don't initialize with README (we already have files)
   - Click "Create repository"

3. **Push to GitHub**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

## Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Wait for Deployment**:
   - Streamlit will automatically install dependencies
   - Your app will be available at a URL like: `https://your-app-name.streamlit.app`

## Step 3: Configuration (Optional)

Create `.streamlit/config.toml` for better performance:
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
enableCORS = false

[browser]
gatherUsageStats = false
```

## Step 4: Update and Redeploy

Whenever you make changes:
```bash
git add .
git commit -m "Update app"
git push origin main
```
Streamlit Cloud automatically redeploys!

## 🎉 Your App is Live!

Your neural data analysis app is now accessible worldwide at your Streamlit Cloud URL.

## Alternative: Heroku Deployment

If you prefer Heroku:

1. **Install Heroku CLI**
2. **Create Procfile**:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

## Need Help?

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io) 