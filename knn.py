import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Đọc dữ liệu
df = pd.read_csv("house_data.csv")

# 2. Chọn đặc trưng và biến mục tiêu
X = df.drop("Price", axis=1)
y = df["Price"]

# Chỉ sử dụng các cột dữ liệu số
X = X.select_dtypes(include=np.number)
X = X.fillna(X.median())

# 3. Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Thử nghiệm các giá trị K
results = []

for k in range(1, 21):
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "K": k,
        "MAE": mae,
        "RMSE": rmse
    })

# 5. Tạo bảng kết quả
results_df = pd.DataFrame(results)
print(results_df)

# 6. Tìm K tốt nhất theo RMSE
best_k = results_df.loc[results_df["RMSE"].idxmin(), "K"]
print("Giá trị K tốt nhất:", int(best_k))

# 7. Trực quan hóa MAE và RMSE
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.lineplot(data=results_df, x="K", y="MAE", marker="o")
plt.title("MAE theo các giá trị K")
plt.xlabel("Giá trị K")
plt.ylabel("MAE")

plt.subplot(1, 2, 2)
sns.lineplot(data=results_df, x="K", y="RMSE", marker="o")
plt.title("RMSE theo các giá trị K")
plt.xlabel("Giá trị K")
plt.ylabel("RMSE")

plt.tight_layout()
plt.show()

# 8. So sánh giá trị thực tế và dự đoán
best_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=int(best_k))
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)

plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title(f"So sánh thực tế và dự đoán (K={int(best_k)})")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "--")

plt.tight_layout()
plt.show()
