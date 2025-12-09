# Real Data Pipeline - Quick Reference

## Quick Start Commands

### Download Real Data
```bash
python scripts/download_real_satellite_data.py --target-count 20
```

### Validate Data Quality
```bash
python scripts/validate_data_quality.py
```

### Prepare Training Data
```bash
# CNN
python scripts/prepare_real_training_data.py

# LSTM
python scripts/prepare_lstm_training_data.py
```

### Train Models
```bash
# CNN
python scripts/train_cnn_on_real_data.py --epochs 50

# LSTM
python scripts/train_lstm_on_real_data.py --epochs 100
```

### Deploy Models
```bash
python scripts/deploy_real_trained_models.py
```

### Compare Performance
```bash
python scripts/compare_model_performance.py
```

---

## Common Troubleshooting Commands

### Check Data Status
```bash
python -c "
from src.database.db_manager import DatabaseManager
db = DatabaseManager('data/agriflux.db')
stats = db.get_database_statistics()
print(f'Real imagery: {stats[\"real_imagery_count\"]}')
print(f'Synthetic imagery: {stats[\"synthetic_imagery_count\"]}')
"
```

### Verify Training Data
```bash
ls -lh data/training/*_real.npy
```

### Check Model Files
```bash
ls -lh models/*_real.pth
```

### View Logs
```bash
# Download logs
tail -f logs/real_data_download.log

# Training logs
tail -f logs/cnn_training.log
tail -f logs/lstm_training.log

# Validation logs
tail -f logs/data_quality_validation.log
```

### Test API Connection
```bash
python -c "
from src.data_processing.sentinel_hub_client import create_client_from_env
client = create_client_from_env()
print('API connection successful!')
"
```

---

## Environment Variables

```bash
# Required
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# Optional
SENTINEL_HUB_BASE_URL=https://services.sentinel-hub.com
USE_AI_MODELS=true
AI_MODEL_PATH=models/crop_health_cnn_real.pth
LSTM_MODEL_PATH=models/crop_health_lstm_real.pth
LOG_LEVEL=INFO
```

---

## File Locations

### Data
- **Processed imagery**: `data/processed/`
- **Training data**: `data/training/`
- **Database**: `data/agriflux.db`

### Models
- **CNN model**: `models/crop_health_cnn_real.pth`
- **LSTM model**: `models/crop_health_lstm_real.pth`
- **Metrics**: `models/*_metrics_real.json`
- **Backups**: `models/backups/`

### Logs
- **Download**: `logs/real_data_download.log`
- **Training**: `logs/cnn_training.log`, `logs/lstm_training.log`
- **Validation**: `logs/data_quality_validation.log`
- **Pipeline**: `logs/pipeline_orchestration.log`

### Reports
- **Comparison**: `reports/model_comparison_report.json`
- **Confusion matrices**: `reports/confusion_matrix_comparison.png`
- **Metrics**: `reports/metrics_comparison.png`

---

## Error Messages and Quick Fixes

| Error | Quick Fix |
|-------|-----------|
| `401 Unauthorized` | Check credentials in `.env` |
| `406 Not Acceptable` | Update to latest API client |
| `429 Rate Limited` | Wait or reduce `--target-count` |
| `No imagery found` | Increase `--cloud-threshold` or `--days-back` |
| `Insufficient training data` | Download more imagery |
| `Model accuracy below threshold` | Download more data or adjust hyperparameters |
| `Out of memory` | Reduce `--batch-size` |
| `Database locked` | Kill conflicting processes |
| `Module not found` | Run `pip install -r requirements.txt` |
| `Permission denied` | Run `chmod -R 755 data/ models/ logs/` |

---

## Performance Benchmarks

### Expected Download Times
- **1 imagery date**: ~30-60 seconds
- **20 imagery dates**: ~10-20 minutes
- **Depends on**: Network speed, API rate limits, region size

### Expected Training Times
- **CNN (50 epochs)**: ~15-30 minutes (GPU), ~2-4 hours (CPU)
- **LSTM (100 epochs)**: ~30-60 minutes (GPU), ~4-8 hours (CPU)
- **Depends on**: Hardware, dataset size, batch size

### Expected Accuracy
- **CNN**: ≥85% validation accuracy
- **LSTM**: ≥80% validation accuracy
- **Improvement over synthetic**: +5-15% accuracy

---

## Data Requirements

### Minimum Requirements
- **Imagery dates**: 15 minimum, 20 recommended
- **Cloud coverage**: <20% per image
- **Temporal span**: 6-12 months
- **Spatial resolution**: 10m (Sentinel-2 bands)

### Storage Requirements
- **Per imagery date**: ~50-100 MB
- **20 dates**: ~1-2 GB
- **Training data**: ~500 MB - 1 GB
- **Models**: ~50-100 MB each
- **Total**: ~3-5 GB for complete pipeline

---

## API Limits

### Sentinel Hub Free Tier
- **Processing units**: 30,000/month
- **Typical usage**: ~100-200 units per imagery date
- **Capacity**: ~150-300 imagery dates/month

### Rate Limits
- **Requests/second**: 10-20 (varies by plan)
- **Automatic retry**: Yes, with exponential backoff
- **Retry-After**: Respected automatically

---

## Validation Checklist

Before training models, verify:

- [ ] At least 15 real imagery dates downloaded
- [ ] All imagery has `synthetic=false` in metadata
- [ ] Cloud coverage <20% for all dates
- [ ] All required bands present (B02, B03, B04, B08)
- [ ] Vegetation indices calculated (NDVI, SAVI, EVI, NDWI)
- [ ] Indices within valid ranges
- [ ] Training data prepared for both CNN and LSTM
- [ ] Class distribution balanced (CNN)
- [ ] Temporal sequences created (LSTM)

After training models, verify:

- [ ] CNN validation accuracy ≥85%
- [ ] LSTM validation accuracy ≥80%
- [ ] Model metadata indicates real data source
- [ ] Confusion matrix shows good performance across all classes
- [ ] Models saved with correct filenames
- [ ] Model registry updated
- [ ] Comparison report shows improvement over synthetic

---

## Support Resources

- **Full Guide**: `docs/REAL_DATA_PIPELINE_GUIDE.md`
- **Logging System**: `docs/LOGGING_SYSTEM.md`
- **Model Deployment**: `docs/MODEL_DEPLOYMENT_GUIDE.md`
- **API Documentation**: https://docs.sentinel-hub.com/
- **Sentinel-2 Info**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
