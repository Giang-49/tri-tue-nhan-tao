# tri-tue-nhan-tao
Hồi quy tuyến tính (Linear Regression) là một trong những mô hình học máy cơ bản và phổ biến nhất trong nhóm học có giám sát (supervised learning), dùng để dự đoán giá trị liên tục (regression).
1. Ý tưởng cốt lõi
Mô hình giả định quan hệ giữa biến đầu vào $X$ và biến mục tiêu $y$ là tuyến tính:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$$

$\beta_0$: hệ số chặn (intercept)
$\beta_1, \beta_2, \dots, \beta_n$: hệ số góc (weights/coefficients)
$\varepsilon$: nhiễu (error)

Mục tiêu của mô hình là tìm bộ tham số $\beta$ sao cho đường thẳng (hoặc siêu phẳng) khớp tốt nhất với dữ liệu.
2. Hàm mất mát (Loss Function)
Thường dùng Mean Squared Error (MSE):
$$J(\beta) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$
Trong đó $\hat{y}_i = X_i \beta$ (dự đoán của mô hình).
3. Cách tìm tham số tối ưu
Có hai cách chính:
a) Phương pháp bình phương tối thiểu (Ordinary Least Squares - OLS)
Công thức đóng (closed-form):
$$\beta = (X^T X)^{-1} X^T y$$
Rất nhanh khi số đặc trưng nhỏ, nhưng tốn kém khi số chiều lớn.
b) Gradient Descent
Cập nhật tham số theo công thức:
$$\beta := \beta - \alpha \frac{\partial J}{\partial \beta}$$

$\alpha$: learning ratePhù hợp với dữ liệu lớn.

4. Các giả định quan trọng của Linear Regression

Quan hệ tuyến tính giữa $X$ và $y$
Các residual (sai số) có phân phối chuẩn, trung bình bằng 0
Phương sai của residual không đổi (homoscedasticity)
Các biến độc lập không tương quan mạnh với nhau (multicollinearity)
Không có outlier quá mạnh ảnh hưởng

Nếu vi phạm nhiều giả định → mô hình dự đoán kém.
5. Đánh giá mô hình

MSE / RMSE: càng nhỏ càng tốt
MAE: trung bình sai số tuyệt đối
R² (Coefficient of Determination): đo mức độ giải thích phương sai của $y$ (0 → 1, càng gần 1 càng tốt)
Adjusted R²: điều chỉnh theo số biến

6. Ứng dụng vào dự đoán giá nhà (House Price Prediction)
Đây là bài toán kinh điển của Linear Regression.
Các bước thực hiện điển hình:

Thu thập dữ liệuVí dụ các đặc trưng phổ biến:
Diện tích (sqft / m²)
Số phòng ngủ, số phòng tắm
Tuổi nhà
Vị trí (khoảng cách đến trung tâm, gần trường học, bệnh viện…)
Tình trạng nhà (mới/cũ, đã sửa chữa…)
Các tiện ích (có gara, hồ bơi, view…)

Tiền xử lý dữ liệu
Xử lý missing values
Xử lý outlier
Chuyển đổi biến phân loại (One-hot Encoding, Ordinal Encoding)
Chuẩn hóa/scale dữ liệu (StandardScaler hoặc MinMaxScaler) — rất quan trọng khi dùng Gradient Descent
Tạo đặc trưng mới (Feature Engineering): ví dụ diện tích mỗi phòng, tỷ lệ phòng ngủ/phòng tắm…

Chia dữ liệu
Train / Validation / Test (thường 70-15-15 hoặc 80-20)

Huấn luyện mô hình
Dùng LinearRegression của scikit-learn (OLS) hoặc tự viết Gradient Descent
Có thể dùng Regularization:
Ridge Regression (L2) → giảm overfitting khi có multicollinearity
Lasso Regression (L1) → tự chọn đặc trưng (feature selection)
Elastic Net (kết hợp L1 + L2)


Đánh giá & tinh chỉnh
Kiểm tra residual plot
Kiểm tra multicollinearity (VIF)
Thử polynomial features nếu quan hệ không hoàn toàn tuyến tính
Cross-validation

Dự đoán
Đưa dữ liệu nhà mới vào mô hình → nhận được giá dự đoán


Ưu điểm khi dùng Linear Regression cho giá nhà

Đơn giản, dễ hiểu, dễ giải thích (có thể nói rõ đặc trưng nào ảnh hưởng mạnh đến giá)
Tốc độ huấn luyện nhanh
Hoạt động tốt khi quan hệ gần tuyến tính và dữ liệu sạch

Nhược điểm

Không nắm bắt được quan hệ phi tuyến tính phức tạp (ví dụ giá nhà tăng phi tuyến theo vị trí)
Nhạy cảm với outlier và multicollinearity
Khi số đặc trưng rất lớn hoặc quan hệ phức tạp → các mô hình như Random Forest, XGBoost, Neural Network thường vượt trội hơn