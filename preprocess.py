import pandas as pd

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Example feature engineering
    df["OvertimeScore"] = df["Overtime"] * 2
    df["PromotionLag"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    df = df.drop(["Overtime", "YearsSinceLastPromotion"], axis=1)

    return df
