# Incremental Mask Metrics Calculation

This document describes the new incremental calculation functionality for mask metrics, which allows for efficient recalculation by only processing frames that haven't been calculated yet or contain invalid/dummy data.

## Overview

The `calculate_mask_metrics_batch()` function has been enhanced to support incremental calculation. When a previous metrics dataframe is provided, the function will:

1. Check which frames already have valid metrics
2. Only calculate metrics for frames that are missing or have invalid data
3. Merge the new results with the existing data
4. Return the combined dataframe

## Key Features

### Smart Frame Detection
- **Missing Frames**: Frames that don't exist in the previous dataframe are automatically calculated
- **Invalid Data Detection**: Frames with dummy/invalid data (zero area, zero perimeter, NaN values) are recalculated
- **Valid Data Preservation**: Frames with valid metrics are preserved and not recalculated

### Validation Criteria
A frame is considered to have valid data if at least one object in that frame has:
- `area > 0`
- `perimeter > 0`
- `circularity` is not NaN
- `orientation` is not NaN

### Efficient Processing
- Only processes frames that actually need calculation
- Maintains the same output format and structure
- Provides detailed logging about which frames are being processed

## Usage

### Basic Usage (No Previous Data)
```python
from utils.mask_metrics import calculate_mask_metrics_batch

# Calculate all metrics
df = calculate_mask_metrics_batch(mask_folder, num_objects=2, object_names=['person', 'background'])
```

### Incremental Usage (With Previous Data)
```python
import pandas as pd

# Load previous metrics
previous_df = pd.read_csv('previous_metrics.csv')

# Calculate only missing/invalid frames
df = calculate_mask_metrics_batch(
    mask_folder, 
    num_objects=2, 
    object_names=['person', 'background'],
    previous_df=previous_df
)
```

### GUI Integration
The `on_export_mask_metrics()` function in the main controller automatically:
1. Checks if a previous metrics file exists
2. Loads and validates the previous data
3. Uses incremental calculation if valid previous data is found
4. Provides user feedback about what was calculated

## Benefits

1. **Performance**: Significantly faster when only a few frames need recalculation
2. **Efficiency**: Avoids redundant processing of already valid data
3. **Robustness**: Handles partial calculations and interrupted processes
4. **User Experience**: Provides clear feedback about what was calculated

## Example Scenarios

### Scenario 1: First Calculation
- No previous file exists
- All frames are calculated
- Full metrics file is created

### Scenario 2: Incremental Addition
- Previous file exists with frames 1-10
- New frames 11-15 are added
- Only frames 11-15 are calculated
- Result combines previous data with new data

### Scenario 3: Invalid Data Correction
- Previous file exists but frames 5-7 have invalid data (zero areas)
- Only frames 5-7 are recalculated
- Other frames are preserved

### Scenario 4: No Changes Needed
- Previous file exists with all valid data
- No calculations are performed
- Previous file is returned unchanged

## Testing

A test script `test_incremental_mask_metrics.py` is provided to verify the functionality:

```bash
python test_incremental_mask_metrics.py
```

The test covers:
- Basic incremental calculation
- Invalid data detection and recalculation
- Partial calculation with missing frames
- Data validation and merging

## Implementation Details

### Function Signature
```python
def calculate_mask_metrics_batch(
    mask_folder: str, 
    num_objects: int = None, 
    object_names: List[str] = None, 
    previous_df: pd.DataFrame = None
) -> pd.DataFrame:
```

### Key Changes
1. Added `previous_df` parameter to accept existing metrics
2. Added frame validation logic to detect invalid data
3. Added smart frame selection to only process needed frames
4. Added dataframe merging logic to combine results
5. Enhanced logging to provide detailed progress information

### Error Handling
- Invalid previous dataframe structure is detected and handled gracefully
- Missing files are handled with appropriate error messages
- Empty results are handled without crashing

## Future Enhancements

Potential improvements could include:
- Support for different validation criteria
- Batch processing optimization
- Progress tracking for long calculations
- Automatic backup of previous files before overwriting 