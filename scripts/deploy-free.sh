#!/bin/bash

# AgriFlux Free Deployment Script
# Choose your free deployment platform

set -e

echo "🌱 AgriFlux Free Deployment Options"
echo "===================================="
echo ""
echo "Choose your FREE deployment platform:"
echo "1. 🚀 Streamlit Community Cloud (Recommended)"
echo "2. 🐙 GitHub Codespaces"
echo "3. 🌐 Render.com"
echo "4. 🚂 Railway.app"
echo "5. 🔥 Google Cloud Run (Free tier)"
echo "6. 📱 Local development server"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "🚀 Setting up Streamlit Community Cloud deployment..."
        echo ""
        echo "📋 Steps to deploy on Streamlit Community Cloud:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://share.streamlit.io"
        echo "3. Connect your GitHub account"
        echo "4. Select this repository"
        echo "5. Set main file path: streamlit_app.py"
        echo "6. Click Deploy!"
        echo ""
        echo "✅ Your app will be live at: https://your-app-name.streamlit.app"
        ;;
    2)
        echo "🐙 Setting up GitHub Codespaces..."
        echo ""
        echo "📋 Steps to use GitHub Codespaces:"
        echo "1. Push your code to GitHub"
        echo "2. Go to your repository on GitHub"
        echo "3. Click 'Code' > 'Codespaces' > 'Create codespace'"
        echo "4. Wait for environment to load"
        echo "5. Run: python run_dashboard.py"
        echo "6. Access via forwarded port 8501"
        echo ""
        echo "✅ Free 60 hours per month!"
        ;;
    3)
        echo "🌐 Setting up Render.com deployment..."
        echo ""
        echo "📋 Steps to deploy on Render.com:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://render.com"
        echo "3. Connect your GitHub account"
        echo "4. Create new Web Service"
        echo "5. Select this repository"
        echo "6. Render will auto-detect the render.yaml config"
        echo "7. Click Deploy!"
        echo ""
        echo "✅ Your app will be live at: https://your-app-name.onrender.com"
        ;;
    4)
        echo "🚂 Setting up Railway.app deployment..."
        echo ""
        echo "📋 Steps to deploy on Railway.app:"
        echo "1. Push your code to GitHub"
        echo "2. Go to https://railway.app"
        echo "3. Connect your GitHub account"
        echo "4. Create new project from GitHub repo"
        echo "5. Railway will auto-detect the railway.json config"
        echo "6. Click Deploy!"
        echo ""
        echo "✅ Your app will be live at: https://your-app-name.railway.app"
        ;;
    5)
        echo "🔥 Setting up Google Cloud Run..."
        echo ""
        echo "📋 Steps to deploy on Google Cloud Run:"
        echo "1. Install Google Cloud CLI"
        echo "2. Run: gcloud auth login"
        echo "3. Run: gcloud config set project YOUR_PROJECT_ID"
        echo "4. Run: gcloud builds submit --config cloudbuild.yaml"
        echo "5. Your app will be deployed automatically"
        echo ""
        echo "✅ Free tier includes 2 million requests per month!"
        ;;
    6)
        echo "📱 Starting local development server..."
        echo ""
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo ""
        echo "🚀 Starting AgriFlux dashboard..."
        python run_dashboard.py
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment setup complete!"
echo "📞 Need help? Contact: support@agriflux.com"