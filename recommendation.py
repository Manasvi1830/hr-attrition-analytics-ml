def generate_recommendations(input_data, risk_score):
    suggestions = []

    if input_data["MonthlyIncome"] < 30000:
        suggestions.append("Consider salary revision")

    if input_data["Overtime"] == 1:
        suggestions.append("Reduce workload or overtime")

    if input_data["YearsSinceLastPromotion"] > 3:
        suggestions.append("Review promotion eligibility")

    if risk_score > 60:
        suggestions.append("Immediate HR intervention required")

    return suggestions
