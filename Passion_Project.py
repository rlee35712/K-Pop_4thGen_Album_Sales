import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def predict_future_sales(data, sales_col, n_future=3):
    x = np.arange(len(data))
    y = data[sales_col].values
    m, b = np.polyfit(x, y, 1)

    future_x = np.arange(len(data), len(data) + n_future)
    future_predictions = m * future_x + b
    future_predictions = np.maximum(future_predictions, 0)  # clamp negatives to 0

    return future_x, future_predictions, m, b

kpop_data = pd.read_csv("kpopfourthgensales.csv")
kpop_data.columns = kpop_data.columns.str.strip()

kpop_data['sales'] = kpop_data['sales'].astype(str).str.replace(",", "", regex=False)
kpop_data['sales'] = pd.to_numeric(kpop_data['sales'], errors="coerce")

print("Your CSV columns are:", list(kpop_data.columns))

print("Which 4th gen K-pop artist would you like to see?")
user_input = input("Izone/Itzy/Stray Kids/Treasure/TXT/(G)I-dle/Cravity/Verivery/Loona/Ateez/Everglow: ").strip()

filtered_data = kpop_data[kpop_data["Artist"].str.contains(user_input, case=False, na=False)]

if filtered_data.empty:
    print(f"No data was found for the group '{user_input}'. Please try again.")
else:
    print(f"Data found for the group '{user_input}'.")
    print("Which type of chart would you like to see for the artist?")
    dv_input = input("Pie chart/Bar chart/Line graph/Predict: ").strip().lower()

    if dv_input == "line graph":
        album_col = 'title'
        sales_col = 'sales'

        for market, group in filtered_data.groupby("country"):
            plt.plot(group[album_col], group[sales_col], marker='o', label=market)

        plt.xlabel(album_col)
        plt.ylabel(sales_col)
        plt.title(f"Sales per Album for {user_input.capitalize()}")
        plt.xticks(rotation=45)
        plt.ylim(0, 600000)
        plt.yticks(range(0, 600001, 60000))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif dv_input == "pie chart":
        sales_col = 'sales'

        pie_data = filtered_data.groupby("country", as_index=False)[sales_col].sum()

        plt.pie(pie_data[sales_col], labels=pie_data["country"], autopct='%1.1f%%')
        plt.title(f"Sales Distribution by Country for {user_input.capitalize()}")
        plt.tight_layout()
        plt.show()

    elif dv_input == "bar chart":
        album_col = 'title'
        sales_col = 'sales'

        pivot_data = filtered_data.pivot(index=album_col, columns="country", values=sales_col)

        pivot_data.plot(kind='bar', color=["#adf7ff", "#00ffc8"])
        plt.xlabel(album_col)
        plt.ylabel(sales_col)
        plt.title(f"Sales per Album for {user_input.capitalize()}")
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
        plt.legend(title="Country")
        plt.tight_layout()
        plt.show()

    elif dv_input == "predict":
        album_col = 'title'
        sales_col = 'sales'

        artist_data = filtered_data.groupby(album_col, as_index=False)[sales_col].sum()

        if len(artist_data) < 2:
            print("Not enough albums to predict a trend.")
        else:
            n_future = 3
            future_x, future_sales, m, b = predict_future_sales(artist_data, sales_col, n_future)

            # Plot actual sales
            x_actual = np.arange(len(artist_data))
            plt.plot(x_actual, artist_data[sales_col], marker='o', label="Actual sales", color="#1f77b4")

            # Plot predicted future sales as dashed line, connected to the last real point
            x_connect = np.concatenate([[x_actual[-1]], future_x])
            y_connect = np.concatenate([[artist_data[sales_col].iloc[-1]], future_sales])
            plt.plot(x_connect, y_connect, marker='o', linestyle='--', label="Predicted", color="#e63946")

            # Label x-axis with album titles + "Future 1", "Future 2", etc.
            all_labels = list(artist_data[album_col]) + [f"Future {i+1}" for i in range(n_future)]
            plt.xticks(np.arange(len(all_labels)), all_labels, rotation=45)

            plt.ylabel(sales_col)
            plt.title(f"Sales Trend & Prediction for {user_input.capitalize()}")
            plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x):,}'))
            plt.legend()
            plt.tight_layout()
            plt.show()

            for i, val in enumerate(future_sales, start=1):
                print(f"Predicted sales for future album #{i}: {val:,.0f}")