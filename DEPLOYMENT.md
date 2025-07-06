# Netflix Movie Recommender - Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Streamlit Cloud (Easiest)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy automatically

### Option 2: Heroku
1. Install Heroku CLI
2. Run these commands:
```bash
heroku create your-app-name
git add .
git commit -m "Deploy Netflix Recommender"
git push heroku main
```

### Option 3: Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Deploy with one click

## 📁 Project Structure
```
Movie-Netflix-Recommendation/
├── app.py                 # Main Streamlit app
├── reco_part.py           # Recommendation engine
├── data/
│   └── netflix_titles_cleaned.csv  # Dataset
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment
├── setup.sh              # Heroku setup script
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🔧 Requirements
- Python 3.8+
- All dependencies in requirements.txt
- Netflix dataset in data/ folder

## 🌐 Access Your App
After deployment, your app will be available at:
- Streamlit Cloud: `https://your-app-name.streamlit.app`
- Heroku: `https://your-app-name.herokuapp.com`
- Railway: `https://your-app-name.railway.app` 