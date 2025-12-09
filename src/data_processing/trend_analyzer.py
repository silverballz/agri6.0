"""
Trend Analyzer - Advanced time series analysis with user-friendly explanations
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

# Optional imports
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Analyze vegetation index trends over time with user-friendly explanations"""
    
    def __init__(self):
        """Initialize the TrendAnalyzer"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - regression analysis will be limited")
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available - seasonal decomposition will be limited")
    
    def fit_regression(self, time_series: pd.Series, index_name: str = 'NDVI') -> Dict:
        """
        Fit linear regression to time series with plain-language interpretation
        
        Args:
            time_series: Pandas Series with time series data
            index_name: Name of the vegetation index
            
        Returns:
            Dictionary with regression results and explanations
        """
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available for regression analysis")
            return self._fallback_regression(time_series, index_name)
        
        if len(time_series) < 2:
            return {
                'slope': 0,
                'intercept': time_series.mean() if len(time_series) > 0 else 0,
                'predictions': time_series.values if len(time_series) > 0 else np.array([]),
                'confidence_lower': time_series.values if len(time_series) > 0 else np.array([]),
                'confidence_upper': time_series.values if len(time_series) > 0 else np.array([]),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"Insufficient data for {index_name} trend analysis.",
                'recommendation': "Collect more data over time to analyze trends.",
                'r_squared': 0
            }
        
        # Prepare data and filter out NaN values
        X = np.arange(len(time_series)).reshape(-1, 1)
        y = time_series.values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y)
        if not valid_mask.any():
            return {
                'slope': 0,
                'intercept': 0,
                'predictions': np.zeros(len(time_series)),
                'confidence_lower': np.zeros(len(time_series)),
                'confidence_upper': np.zeros(len(time_series)),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"No valid data available for {index_name} trend analysis.",
                'recommendation': "Check data quality and ensure valid measurements are available.",
                'r_squared': 0
            }
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) < 2:
            return {
                'slope': 0,
                'intercept': y_valid.mean() if len(y_valid) > 0 else 0,
                'predictions': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'confidence_lower': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'confidence_upper': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"Insufficient valid data for {index_name} trend analysis.",
                'recommendation': "Collect more data over time to analyze trends.",
                'r_squared': 0
            }
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_valid, y_valid)
        
        # Calculate predictions for all points
        predictions = model.predict(X)
        
        # Calculate confidence intervals (using residual standard error from valid points only)
        predictions_valid = model.predict(X_valid)
        residuals = y_valid - predictions_valid
        std_error = np.std(residuals)
        confidence_interval = 1.96 * std_error  # 95% CI
        
        # Calculate rate of change in user-friendly units
        slope = model.coef_[0]
        days_in_data = len(time_series)
        
        # Convert to percentage change per week and month (using valid data mean)
        mean_value = y_valid.mean()
        if mean_value != 0 and not np.isnan(mean_value):
            weekly_change = (slope * 7 / mean_value) * 100
            monthly_change = (slope * 30 / mean_value) * 100
        else:
            weekly_change = 0
            monthly_change = 0
        
        # Calculate R-squared using valid data only
        r_squared = model.score(X_valid, y_valid)
        
        # Generate plain-language explanation
        explanation, recommendation = self._generate_trend_explanation(
            weekly_change, monthly_change, r_squared, index_name
        )
        
        return {
            'slope': slope,
            'intercept': model.intercept_,
            'predictions': predictions,
            'confidence_lower': predictions - confidence_interval,
            'confidence_upper': predictions + confidence_interval,
            'weekly_change_pct': weekly_change,
            'monthly_change_pct': monthly_change,
            'explanation': explanation,
            'recommendation': recommendation,
            'r_squared': r_squared
        }
    
    def _fallback_regression(self, time_series: pd.Series, index_name: str) -> Dict:
        """Fallback regression using numpy when sklearn is unavailable"""
        if len(time_series) < 2:
            return {
                'slope': 0,
                'intercept': time_series.mean() if len(time_series) > 0 else 0,
                'predictions': time_series.values if len(time_series) > 0 else np.array([]),
                'confidence_lower': time_series.values if len(time_series) > 0 else np.array([]),
                'confidence_upper': time_series.values if len(time_series) > 0 else np.array([]),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"Insufficient data for {index_name} trend analysis.",
                'recommendation': "Collect more data over time to analyze trends.",
                'r_squared': 0
            }
        
        # Simple linear regression using numpy
        x = np.arange(len(time_series))
        y = time_series.values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y)
        if not valid_mask.any():
            return {
                'slope': 0,
                'intercept': 0,
                'predictions': np.zeros(len(time_series)),
                'confidence_lower': np.zeros(len(time_series)),
                'confidence_upper': np.zeros(len(time_series)),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"No valid data available for {index_name} trend analysis.",
                'recommendation': "Check data quality and ensure valid measurements are available.",
                'r_squared': 0
            }
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 2:
            return {
                'slope': 0,
                'intercept': y_valid.mean() if len(y_valid) > 0 else 0,
                'predictions': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'confidence_lower': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'confidence_upper': np.full(len(time_series), y_valid.mean() if len(y_valid) > 0 else 0),
                'weekly_change_pct': 0,
                'monthly_change_pct': 0,
                'explanation': f"Insufficient valid data for {index_name} trend analysis.",
                'recommendation': "Collect more data over time to analyze trends.",
                'r_squared': 0
            }
        
        # Calculate slope and intercept using valid data
        slope = np.cov(x_valid, y_valid)[0, 1] / np.var(x_valid) if np.var(x_valid) != 0 else 0
        intercept = y_valid.mean() - slope * x_valid.mean()
        
        # Predictions for all points
        predictions = slope * x + intercept
        
        # Simple confidence interval (using valid data only)
        predictions_valid = slope * x_valid + intercept
        residuals = y_valid - predictions_valid
        std_error = np.std(residuals)
        confidence_interval = 1.96 * std_error
        
        # Calculate percentage changes (using valid data mean)
        mean_value = y_valid.mean()
        if mean_value != 0 and not np.isnan(mean_value):
            weekly_change = (slope * 7 / mean_value) * 100
            monthly_change = (slope * 30 / mean_value) * 100
        else:
            weekly_change = 0
            monthly_change = 0
        
        # Calculate R-squared using valid data only
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        explanation, recommendation = self._generate_trend_explanation(
            weekly_change, monthly_change, r_squared, index_name
        )
        
        return {
            'slope': slope,
            'intercept': intercept,
            'predictions': predictions,
            'confidence_lower': predictions - confidence_interval,
            'confidence_upper': predictions + confidence_interval,
            'weekly_change_pct': weekly_change,
            'monthly_change_pct': monthly_change,
            'explanation': explanation,
            'recommendation': recommendation,
            'r_squared': r_squared
        }
    
    def _generate_trend_explanation(
        self, 
        weekly_change: float, 
        monthly_change: float, 
        r_squared: float,
        index_name: str
    ) -> Tuple[str, str]:
        """Generate plain-language explanation and recommendation"""
        
        # Determine trend strength
        if r_squared > 0.7:
            strength = "strong"
        elif r_squared > 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        # Generate explanation
        if abs(weekly_change) < 0.5:
            explanation = f"Your {index_name} is stable with minimal change over time (less than 0.5% per week)."
            recommendation = "Continue current management practices. Monitor regularly for any changes."
        elif weekly_change > 0:
            explanation = f"Your crops are improving by {abs(weekly_change):.1f}% per week ({abs(monthly_change):.1f}% per month). This is a {strength} trend."
            if weekly_change > 2:
                recommendation = "Excellent progress! Maintain current irrigation and fertilization practices."
            else:
                recommendation = "Good progress. Continue monitoring and maintain current practices."
        else:
            explanation = f"Your crops are declining by {abs(weekly_change):.1f}% per week ({abs(monthly_change):.1f}% per month). This is a {strength} trend."
            if abs(weekly_change) > 2:
                recommendation = "⚠️ Significant decline detected. Consider increasing irrigation, investigating pest/disease issues, or adjusting fertilization."
            else:
                recommendation = "⚠️ Moderate decline detected. Monitor closely and consider adjusting management practices."
        
        return explanation, recommendation
    
    def detect_anomalies(
        self, 
        time_series: pd.Series, 
        dates: pd.DatetimeIndex, 
        threshold_std: float = 2.0
    ) -> Dict:
        """
        Detect anomalies with plain-language descriptions
        
        Args:
            time_series: Pandas Series with time series data
            dates: DatetimeIndex with corresponding dates
            threshold_std: Number of standard deviations for anomaly threshold
            
        Returns:
            Dictionary with anomaly information and descriptions
        """
        if len(time_series) < 3:
            return {
                'anomalies': np.array([False] * len(time_series)),
                'z_scores': np.zeros(len(time_series)),
                'descriptions': [],
                'count': 0
            }
        
        # Calculate z-scores
        mean = time_series.mean()
        std = time_series.std()
        
        # Use tolerance check for std close to 0 (handles floating point precision)
        if std < 1e-10:
            return {
                'anomalies': np.array([False] * len(time_series)),
                'z_scores': np.zeros(len(time_series)),
                'descriptions': [],
                'count': 0
            }
        
        z_scores = np.abs((time_series - mean) / std)
        anomalies = z_scores > threshold_std
        
        # Generate descriptions for each anomaly
        anomaly_descriptions = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                date = dates[i]
                value = time_series.iloc[i]
                deviation = z_scores.iloc[i] if hasattr(z_scores, 'iloc') else z_scores[i]
                
                if value > mean:
                    direction = "spike"
                    action = "Verify data quality or investigate unusual growth conditions (e.g., recent rainfall, fertilization)"
                else:
                    direction = "drop"
                    action = "Investigate irrigation issues, pest damage, disease, or environmental stress"
                
                description = (
                    f"Unusual {direction} detected on {date.strftime('%b %d, %Y')}: "
                    f"{value:.3f} ({deviation:.1f}σ from normal). {action}."
                )
                
                anomaly_descriptions.append({
                    'date': date,
                    'value': value,
                    'z_score': deviation,
                    'description': description,
                    'direction': direction
                })
        
        return {
            'anomalies': anomalies,
            'z_scores': z_scores,
            'descriptions': anomaly_descriptions,
            'count': int(sum(anomalies))
        }
    
    def decompose_seasonal(
        self, 
        time_series: pd.Series, 
        period: int = 365
    ) -> Optional[Dict]:
        """
        Decompose time series with explanations for each component
        
        Args:
            time_series: Pandas Series with time series data
            period: Period for seasonal decomposition (default: 365 days)
            
        Returns:
            Dictionary with decomposition results and explanations, or None if unavailable
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available for seasonal decomposition")
            return None
        
        if len(time_series) < 2 * period:
            logger.warning(f"Insufficient data for seasonal decomposition (need at least {2 * period} points)")
            return None
        
        try:
            # Perform seasonal decomposition
            result = seasonal_decompose(
                time_series,
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            
            # Generate explanations
            trend_direction = "increasing" if result.trend.iloc[-1] > result.trend.iloc[0] else "decreasing"
            seasonal_amplitude = result.seasonal.max() - result.seasonal.min()
            
            # Calculate residual statistics
            residual_std = result.resid.std()
            residual_max = result.resid.abs().max()
            
            explanations = {
                'trend': (
                    f"The overall {trend_direction} trend shows the long-term direction of vegetation health, "
                    f"removing seasonal and random variations. This represents the underlying growth pattern."
                ),
                'seasonal': (
                    f"Seasonal patterns repeat annually with amplitude of {seasonal_amplitude:.3f}. "
                    f"This represents expected seasonal growth cycles (e.g., monsoon, winter dormancy)."
                ),
                'residual': (
                    f"Residuals show unexplained variations (std: {residual_std:.3f}, max: {residual_max:.3f}) "
                    f"after removing trend and seasonality. Large residuals may indicate unusual events requiring investigation."
                )
            }
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'explanations': explanations,
                'seasonal_amplitude': seasonal_amplitude,
                'trend_direction': trend_direction
            }
        
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
            return None
    
    def calculate_rate_of_change(
        self, 
        time_series: pd.Series, 
        dates: pd.DatetimeIndex, 
        window: int = 7
    ) -> Dict:
        """
        Calculate rate of change with comparison to historical averages
        
        Args:
            time_series: Pandas Series with time series data
            dates: DatetimeIndex with corresponding dates
            window: Window size for rate calculation (default: 7 days)
            
        Returns:
            Dictionary with rate of change information
        """
        if len(time_series) < window + 1:
            return {
                'rate': pd.Series([0] * len(time_series)),
                'rate_pct_weekly': pd.Series([0] * len(time_series)),
                'growth_periods': pd.Series([False] * len(time_series)),
                'decline_periods': pd.Series([False] * len(time_series)),
                'avg_growth_rate': 0,
                'avg_decline_rate': 0,
                'historical_avg_rate': 0,
                'significant_changes': []
            }
        
        # Calculate rate of change
        rate = time_series.diff(window) / window
        
        # Convert to percentage change per week
        rate_pct_weekly = (rate / time_series) * 100
        
        # Replace inf and nan values
        rate_pct_weekly = rate_pct_weekly.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Classify as growth or decline
        growth_periods = rate > 0.01
        decline_periods = rate < -0.01
        
        # Calculate historical average rate
        historical_avg_rate = rate.mean()
        
        # Identify significant deviations
        rate_std = rate.std()
        significant_changes = []
        
        if rate_std > 0:
            for i, r in enumerate(rate):
                if not np.isnan(r) and abs(r - historical_avg_rate) > 2 * rate_std:
                    date = dates[i]
                    pct_change = rate_pct_weekly.iloc[i]
                    
                    if r > historical_avg_rate:
                        description = (
                            f"Rapid growth on {date.strftime('%b %d')}: {pct_change:.1f}% per week "
                            f"(above normal by {abs(r - historical_avg_rate):.4f})"
                        )
                    else:
                        description = (
                            f"Rapid decline on {date.strftime('%b %d')}: {pct_change:.1f}% per week "
                            f"(below normal by {abs(r - historical_avg_rate):.4f})"
                        )
                    
                    significant_changes.append({
                        'date': date,
                        'rate': r,
                        'rate_pct_weekly': pct_change,
                        'description': description
                    })
        
        # Calculate average rates for growth and decline periods
        avg_growth_rate = rate[growth_periods].mean() if growth_periods.any() else 0
        avg_decline_rate = rate[decline_periods].mean() if decline_periods.any() else 0
        
        return {
            'rate': rate,
            'rate_pct_weekly': rate_pct_weekly,
            'growth_periods': growth_periods,
            'decline_periods': decline_periods,
            'avg_growth_rate': avg_growth_rate,
            'avg_decline_rate': avg_decline_rate,
            'historical_avg_rate': historical_avg_rate,
            'significant_changes': significant_changes
        }
    
    def generate_day_wise_comparison(
        self, 
        time_series_data: pd.DataFrame, 
        date1: datetime, 
        date2: datetime
    ) -> Optional[Dict]:
        """
        Generate day-wise comparison between two dates
        
        Args:
            time_series_data: DataFrame with time series data
            date1: First date for comparison
            date2: Second date for comparison
            
        Returns:
            Dictionary with comparison results, or None if data unavailable
        """
        # Filter data for the two dates
        data1 = time_series_data[time_series_data['date'] == date1]
        data2 = time_series_data[time_series_data['date'] == date2]
        
        if data1.empty or data2.empty:
            return None
        
        # Calculate changes for each index
        changes = {}
        for index in ['NDVI', 'SAVI', 'EVI', 'NDWI']:
            if index in data1.columns and index in data2.columns:
                val1 = data1[index].values[0]
                val2 = data2[index].values[0]
                delta = val2 - val1
                pct_change = (delta / val1 * 100) if val1 != 0 else 0
                
                changes[index] = {
                    'value1': val1,
                    'value2': val2,
                    'delta': delta,
                    'pct_change': pct_change,
                    'interpretation': self._interpret_change(index, delta, pct_change)
                }
        
        return changes
    
    def _interpret_change(self, index: str, delta: float, pct_change: float) -> str:
        """Interpret the meaning of a change in vegetation index"""
        if abs(pct_change) < 2:
            return "Minimal change - vegetation stable"
        elif delta > 0:
            if pct_change > 10:
                return "Significant improvement - excellent growth"
            else:
                return "Moderate improvement - positive trend"
        else:
            if pct_change < -10:
                return "⚠️ Significant decline - immediate attention needed"
            else:
                return "⚠️ Moderate decline - monitor closely"
