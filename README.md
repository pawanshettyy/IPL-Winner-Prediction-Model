# IPL Match Predictor

## Overview
This machine learning project predicts the outcome of Indian Premier League (IPL) cricket matches using historical match data. It combines statistical analysis with machine learning techniques to forecast match winners with competitive accuracy.

## Features
- **Match Outcome Prediction**: Predict the winning team of an IPL match based on team statistics, venue information, and toss details
- **Feature Engineering**: Comprehensive feature generation including:
  - Team win rates
  - Venue-specific performance
  - Head-to-head records
  - Toss advantage analysis
  - Home ground advantage
- **Multiple Models**: Implements and compares Random Forest and XGBoost classifiers
- **Hyperparameter Tuning**: Uses GridSearchCV for optimal model configuration
- **Model Persistence**: Saves trained models and transformers for easy reuse
- **Simple Prediction Interface**: Straightforward function for making predictions on new matches

## Requirements
- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle

## Data Requirements
The project requires two CSV files:
1. `matches.csv` - Historical IPL match data
2. `deliveries.csv` - Ball-by-ball data for IPL matches

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ipl-match-predictor.git
cd ipl-match-predictor
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

3. Place your `matches.csv` and `deliveries.csv` files in the project directory

## Usage

### Training the Model
Run the main script to train the models:
```bash
python ipl_predictor.py
```

This will:
1. Load and preprocess the data
2. Engineer features from match statistics
3. Train Random Forest and XGBoost models
4. Perform hyperparameter tuning
5. Save the best-performing model

### Making Predictions
After training, you can use the `predict_match_winner` function to predict outcomes for new matches:

```python
from ipl_predictor import predict_match_winner

predicted_winner, confidence = predict_match_winner(
    team1='Mumbai Indians',
    team2='Chennai Super Kings',
    venue='Wankhede Stadium',
    city='Mumbai',
    toss_winner='Mumbai Indians',
    toss_decision='bat'
)

print(f"Predicted winner: {predicted_winner} with {confidence:.2f}% confidence")
```

## Model Details

### Feature Set
The model leverages these key features:
- Team win rates
- Win rate differential
- Venue-specific team performance
- Toss advantage metrics
- Head-to-head performance
- Home ground advantage
- Encoded categorical variables (teams, venues, cities)
- Toss decision impact (bat/field)

### Performance
The model achieves competitive prediction accuracy, with performance metrics displayed during training. The final model selection is based on test set accuracy.

## Files
- `ipl_predictor.py` - Main script containing the data processing pipeline and model training code
- `best_ipl_predictor_[model_name].pkl` - Saved best model (either Random Forest or XGBoost)
- `feature_encoders.pkl` - Saved label encoders for categorical features
- `feature_scaler.pkl` - Saved feature scaler for numerical features

## Future Improvements
- Adding player-specific statistics
- Incorporating weather data
- Updating with recent match data
- Implementing a web interface for predictions
- Expanding to other cricket leagues

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Data sourced from publicly available IPL match statistics
- Built with scikit-learn and XGBoost libraries
