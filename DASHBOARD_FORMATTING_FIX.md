# Dashboard Formatting Fix

## Issue
The Model Performance page was crashing with the error:
```
ValueError: Unknown format code 'd' for object of type 'float'
```

## Root Cause
In the CNN model metrics JSON file (`models/cnn_model_metrics_real.json`), the classification report contains support values as floats (e.g., `382.0`) rather than integers. When the dashboard tried to format these values using the integer format code `{:d}`, it failed because Python's format function requires integers for the 'd' format specifier.

### Location of Error
**File**: `src/dashboard/pages/model_performance.py`
**Line**: 384
**Function**: `display_cnn_performance()`

### Code That Failed
```python
# When parsing classification report from dict
class_data.append({
    'Class': class_name,
    'Precision': class_metrics.get('precision', 0),
    'Recall': class_metrics.get('recall', 0),
    'F1-Score': class_metrics.get('f1-score', 0),
    'Support': class_metrics.get('support', 0)  # ❌ Returns float
})

# Later, when formatting
df.style.format({
    'Precision': '{:.2f}',
    'Recall': '{:.2f}',
    'F1-Score': '{:.2f}',
    'Support': '{:d}'  # ❌ Expects integer, gets float
})
```

## Solution
Convert the support value to an integer when extracting from the dictionary:

```python
# Fixed code
class_data.append({
    'Class': class_name,
    'Precision': class_metrics.get('precision', 0),
    'Recall': class_metrics.get('recall', 0),
    'F1-Score': class_metrics.get('f1-score', 0),
    'Support': int(class_metrics.get('support', 0))  # ✅ Convert to int
})
```

## Files Modified
- ✅ `src/dashboard/pages/model_performance.py` (line 358)

## Testing
Created and ran `test_classification_report_fix.py` to verify the fix:
- ✅ Successfully formats support values as integers
- ✅ Handles float input correctly
- ✅ No errors when applying style formatting

## Impact
- **Before**: Dashboard crashed when viewing CNN model performance
- **After**: Dashboard displays classification report correctly with proper formatting

## Related Context
This issue only appeared after implementing the real data pipeline because:
1. The real-trained models use PyTorch's classification_report output
2. PyTorch returns support values as floats (e.g., 382.0)
3. The synthetic models may have had integer support values
4. The dashboard code assumed support would always be integers

## Prevention
The fix ensures type safety by explicitly converting to int, which will work regardless of whether the source data is float or int.
