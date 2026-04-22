from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# =========================
# 🧠 AI MODEL FUNCTION
# =========================
def analyze_stock(stock):
    try:
        # 📊 Get 1 year data
        data = yf.download(stock, period="1y")

        # Fix MultiIndex issue
        if not data.empty and hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)

        # Safety check
        if data.empty or len(data) < 2:
            return None

        data.dropna(inplace=True)

        # =========================
        # 📌 CLOSE PRICE
        # =========================
        close_prices = data["Close"]

        # =========================
        # 📊 FEATURES
        # =========================
        data["Day"] = np.arange(len(data))
        data["MA50"] = close_prices.rolling(50).mean()
        data["MA200"] = close_prices.rolling(200).mean()

        # =========================
        # 🤖 AI MODEL
        # =========================
        X = data["Day"].values.reshape(-1, 1)
        y = close_prices.values

        model = LinearRegression()
        model.fit(X, y)

        next_day = np.array([[len(data) + 1]])
        predicted_price = model.predict(next_day).item()

        # =========================
        # 📊 METRICS
        # =========================
        current_price = float(close_prices.iloc[-1])
        yesterday_close = float(close_prices.iloc[-2])
        first_price = float(close_prices.iloc[0])

        daily_change = current_price - yesterday_close
        profit_loss = ((current_price - first_price) / first_price) * 100

        signal = "BUY 📈" if predicted_price > current_price else "SELL 📉"

        # =========================
        # 📈 CHART FUNCTION
        # =========================
        def get_chart(period, interval):
            df = yf.download(stock, period=period, interval=interval)

            if not df.empty and hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)

            return list(df["Close"].dropna().values) if not df.empty else []

        chart_1d = get_chart("1d", "5m")
        chart_1w = get_chart("5d", "30m")
        chart_1m = get_chart("1mo", "1d")
        chart_1y = list(close_prices.dropna().values)

        return {
            "current_price": round(current_price, 2),
            "yesterday_close": round(yesterday_close, 2),
            "daily_change": round(daily_change, 2),
            "profit_loss": round(profit_loss, 2),
            "predicted_price": round(predicted_price, 2),
            "signal": signal,
            "ma50": round(float(data["MA50"].iloc[-1]), 2),
            "ma200": round(float(data["MA200"].iloc[-1]), 2),

            "chart_1d": chart_1d,
            "chart_1w": chart_1w,
            "chart_1m": chart_1m,
            "chart_1y": chart_1y,
        }

    except Exception as e:
        print("Error:", e)
        return None


# =========================
# 🌐 ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stocks")
def stocks():
    return render_template("stocks.html")


@app.route("/dashboard", methods=["POST"])
def dashboard():
    stock = request.form.get("stock")

    result = analyze_stock(stock)

    if result is None:
        return "Error: Invalid stock or no data."

    return render_template(
        "dashboard.html",
        stock=stock,
        current=result["current_price"],
        yesterday=result["yesterday_close"],
        change=result["daily_change"],
        profit=result["profit_loss"],
        predicted=result["predicted_price"],
        signal=result["signal"],
        ma50=result["ma50"],
        ma200=result["ma200"],
        chart_1d=result["chart_1d"],
        chart_1w=result["chart_1w"],
        chart_1m=result["chart_1m"],
        chart_1y=result["chart_1y"],
    )


# =========================
# 🚀 RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)