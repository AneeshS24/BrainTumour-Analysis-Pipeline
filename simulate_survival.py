import pandas as pd

def simulate_survival(classification_csv, risk_csv, output_csv):
    # Load classification predictions
    classification_df = pd.read_csv(classification_csv)
    print("Classification CSV columns:", classification_df.columns.tolist())

    # Load risk labels
    risk_df = pd.read_csv(risk_csv)
    print("Risk CSV columns:", risk_df.columns.tolist())

    # Merge both DataFrames on the image column (case-sensitive fix)
    merged_df = pd.merge(classification_df, risk_df, left_on="Image", right_on="image", how="inner")

    # Optional: drop duplicate 'image' column (since 'Image' already exists)
    merged_df.drop(columns=["image"], inplace=True)

    # Simulate survival days based on risk and tumor type
    def estimate_days(row):
        base = 180  # base survival in days
        if row["risk"] == "High":
            base -= 90
        elif row["risk"] == "Medium":
            base -= 45
        else:
            base += 30

        if row["Prediction"] == "Glioma":
            base -= 30
        elif row["Prediction"] == "Pituitary":
            base += 20
        elif row["Prediction"] == "Meningioma":
            base -= 10

        return max(base, 30)  # Ensure survival is at least 30 days

    # Apply survival estimation
    merged_df["estimated_survival_days"] = merged_df.apply(estimate_days, axis=1)

    # Save results
    merged_df.to_csv(output_csv, index=False)
    print(f"\nSurvival estimation saved to: {output_csv}")

if __name__ == "__main__":
    classification_csv = "vit_predictions.csv"
    risk_csv = "risk_labeled_data.csv"
    output_csv = "survival_estimations.csv"

    simulate_survival(classification_csv, risk_csv, output_csv)
