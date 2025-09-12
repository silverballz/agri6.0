# 🚀 Deploy AgriFlux to Render

## Quick Deploy Steps

### 1. Push to GitHub (if not already done)
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. Deploy on Render
1. Go to **https://render.com**
2. Sign up/Login with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repository
5. Select the `agriflux` repository
6. Render will auto-detect the `render.yaml` configuration

### 3. Configuration (Auto-detected)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
- **Environment**: Python
- **Plan**: Free

### 4. Environment Variables (Optional)
Add these in Render dashboard if needed:
- `ENVIRONMENT=production`
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### 5. Deploy!
Click **"Create Web Service"** and wait for deployment (5-10 minutes)

## Your App Will Be Live At:
`https://your-app-name.onrender.com`

## Troubleshooting

### If Build Fails:
1. Check build logs in Render dashboard
2. Ensure `requirements.txt` has all dependencies
3. Verify `streamlit_app.py` is in root directory

### If App Crashes:
1. Check runtime logs in Render dashboard
2. Look for import errors or missing modules
3. Ensure all Python files have proper imports

### Common Issues:
- **Port binding**: Render automatically sets `$PORT` environment variable
- **File paths**: Use relative paths from project root
- **Dependencies**: Keep requirements.txt minimal for faster builds

## Success! 🎉
Your AgriFlux platform will be accessible worldwide with:
- ✅ Free hosting
- ✅ HTTPS SSL certificate
- ✅ Auto-deploy on git push
- ✅ Custom domain support (paid plans)