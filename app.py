from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    input_data = None
    recommendation = None
    if request.method == "POST":
        input_data = {
            "gender": request.form["gender"],
            "age": int(request.form["age"]),
            "country": request.form["country"],
            "city": request.form["city"],
            "customer_segment": request.form["customer_segment"],
            "tenure_months": int(request.form["tenure_months"]),
            "signup_channel": request.form["signup_channel"],
            "contract_type": request.form["contract_type"],
            "monthly_logins": int(request.form["monthly_logins"]),
            "weekly_active_days": int(request.form["weekly_active_days"]),
            "avg_session_time": float(request.form["avg_session_time"]),
            "features_used": int(request.form["features_used"]),
            "usage_growth_rate": float(request.form["usage_growth_rate"]),
            "last_login_days_ago": int(request.form["last_login_days_ago"]),
            "monthly_fee": float(request.form["monthly_fee"]),
            "total_revenue": float(request.form["total_revenue"]),
            "payment_method": request.form["payment_method"],
            "payment_failures": int(request.form["payment_failures"]),
            "discount_applied": request.form["discount_applied"],
            "price_increase_last_3m": request.form["price_increase_last_3m"],
            "support_tickets": int(request.form["support_tickets"]),
            "avg_resolution_time": float(request.form["avg_resolution_time"]),
            "complaint_type": request.form["complaint_type"],
            "csat_score": float(request.form["csat_score"]),
            "escalations": int(request.form["escalations"]),
            "email_open_rate": float(request.form["email_open_rate"]),
            "marketing_click_rate": float(request.form["marketing_click_rate"]),
            "nps_score": int(request.form["nps_score"]),
            "survey_response": request.form["survey_response"],
            "referral_count": int(request.form["referral_count"])
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(columns=columns, fill_value=0)

        churn_probability = model.predict_proba(input_df)[0][1]

        if churn_probability > 0.2:
            prediction = "Likely to Churn"
            recommendation = "Recommended action: Contact the customer, offer support, or provide a retention discount."
        else:
            prediction = "Likely to Stay"
            recommendation = "Recommended action: Continue normal engagement and monitor customer satisfaction."

        probability = round(churn_probability * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        recommendation=recommendation,
        form_data=input_data
    )

if __name__ == "__main__":
    app.run(debug=True)