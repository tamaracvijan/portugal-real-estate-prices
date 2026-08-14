# Portugal Real Estate Price Prediction

Data cleaning, exploratory analysis, and price prediction on a 2024 Portuguese real estate dataset (Python, pandas, scikit-learn, XGBoost).

## Dataset
[Portugal Real Estate 2024](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024) from Kaggle - 174,727 property listings with 23 original features (price, area, location, rooms, energy certificate, etc.).

## Workflow

### 1. Data cleaning
- Dropped rows with missing target values (`Price`) and columns with very few missing values (`Town`, `Type`, `EnergyCertificate`, `Parking`, `Elevator`)
- Filled missing numeric values (`TotalArea`, `NumberOfBathrooms`, `ConstructionYear`, `TotalRooms`, `LivingArea`) using the median, grouped by property `Type`
- Filled the categorical `EnergyEfficiencyLevel` using the most common value (mode), grouped by property `Type`
- Filled `Garage` and `ElectricCarsCharging` with `False`, assuming missing means the feature is absent
- Dropped columns with 100,000+ missing values (`Floor`, `ConservationStatus`, `LotSize`, `PublishDate`) and redundant columns (`BuiltArea`, `NumberOfWC`, `NumberOfBedrooms`)

### 2. Handling data errors
Found physically impossible values (negative area, negative bathroom counts, area in the billions of square meters), removed these, then applied the IQR method (with a wider multiplier to preserve legitimate large properties like villas and farms) to remove remaining outliers.

### 3. Encoding
- **Target encoding** for `Town` (2,314 unique values) - replaced each town with its average price from the training set, to avoid the sparse columns one-hot encoding would create
- **One-hot encoding** for the remaining categorical columns (`District`, `City`, `Type`, `EnergyCertificate`, `EnergyEfficiencyLevel`)

### 4. Modeling
Trained and compared three models:

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 202,706 | 0.594 |
| Random Forest | **143,455** | **0.797** |
| XGBoost (tuned) | 147,766 | 0.784 |

**Random Forest** performed best and was selected as the final model.

## Key findings
- `Town` (target-encoded) and `NumberOfBathrooms` were by far the most important features, together accounting for over 57% of the model's decisions
- Price does not depend on features in a simple linear way - Random Forest and XGBoost both significantly outperformed Linear Regression

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (Linear Regression, Random Forest, preprocessing)
- XGBoost
- Matplotlib

## Next steps
- Hyperparameter tuning with GridSearchCV
- Additional feature engineering (e.g. price per square meter)
- Testing other models (e.g. LightGBM)
