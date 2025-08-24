import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def ema(series, alpha=0.2):
    """Exponential moving average smoothing"""
    return series.ewm(alpha=alpha, adjust=False).mean()

def generate_alerts(df, warning_th=50, critical_th=70, cusum_th=200, delta_th=15,
                    forecast_window=10, forecast_horizon=5):
    """Generate anomaly alerts and predictive risk signals"""

    # Smooth anomaly score
    df["score_ema"] = ema(df["Abnormality_score"], alpha=0.2)

    # Velocity (rate of change)
    df["score_delta"] = df["Abnormality_score"].diff().fillna(0)

    # CUSUM (cumulative stress beyond baseline of 40)
    df["score_cusum"] = (df["Abnormality_score"] - 40).clip(lower=0).cumsum()

    # Basic alert rules
    def classify_alert(row):
        if row["score_ema"] >= critical_th or row["score_delta"] >= delta_th or row["score_cusum"] >= cusum_th:
            return "CRITICAL"
        elif row["score_ema"] >= warning_th:
            return "WARNING"
        else:
            return "OK"

    df["alert_state"] = df.apply(classify_alert, axis=1)

    # Forecast future anomaly score
    forecast = []
    for i in range(len(df)):
        if i < forecast_window:
            forecast.append(np.nan)
            continue
        X = np.arange(forecast_window).reshape(-1, 1)
        y = df["Abnormality_score"].iloc[i-forecast_window:i].values
        model = LinearRegression().fit(X, y)
        y_pred = model.predict([[forecast_window + forecast_horizon]])[0]
        forecast.append(y_pred)

    df["forecast_score"] = forecast

    # Predictive alert
    df["predictive_alert"] = df["forecast_score"].apply(
        lambda x: "PREDICTED_RISK" if pd.notna(x) and x >= critical_th else "SAFE"
    )

    return df


def main(input_csv, output_csv, timestamp_col="Time", plot_graph=True):
    # Load CSV
    df = pd.read_csv(input_csv, parse_dates=[timestamp_col])

    if "Abnormality_score" not in df.columns:
        raise ValueError("CSV must contain 'Abnormality_score' column!")

    # Generate alerts
    df = generate_alerts(df)

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Alerts saved to {output_csv}")

    # Optional visualization
    if plot_graph:
        plt.figure(figsize=(12, 6))
        plt.plot(df[timestamp_col], df["Abnormality_score"], label="Raw Score", alpha=0.5)
        plt.plot(df[timestamp_col], df["score_ema"], label="EMA Smoothed", linewidth=2)
        plt.plot(df[timestamp_col], df["forecast_score"], label="Forecasted", linestyle="--")

        plt.axhline(50, color="orange", linestyle="--", label="Warning Threshold (50)")
        plt.axhline(70, color="red", linestyle="--", label="Critical Threshold (70)")

        plt.title("Anomaly Scores with Alerts & Forecasts")
        plt.xlabel("Time")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate alerts from anomaly scores CSV")
    parser.add_argument("--input_csv", required=True, help="Path to input anomaly score CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV with alerts")
    parser.add_argument("--timestamp_col", default="Time", help="Name of timestamp column")
    parser.add_argument("--no_plot", action="store_true", help="Disable graph plotting")

    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.timestamp_col, not args.no_plot)
