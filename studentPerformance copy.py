import kagglehub

# Download latest version
path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")

import streamlit as st
import pandas as pd

# Tải dữ liệu từ file CSV 
df = pd.read_csv(r"CSA\L10\CP03\student_data.csv")  

# Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu
st.write("📊 Dữ liệu ban đầu:", df.head())
df.set_index("student_id", inplace=True)

# Kiểm tra xem có giá trị nào bị thiếu không
df_nulls = df.isnull().sum().to_frame(name="Số giá trị thiếu")
st.write("📉 Tổng số giá trị thiếu theo cột:")
st.write(df_nulls)

# Xóa các dòng có giá trị thiếu để đảm bảo phân tích chính xác
df_clean = df.dropna()

# Hiển thị lại dữ liệu sau khi làm sạch
st.write("✅ Dữ liệu sau khi làm sạch:", df_clean.head())

import matplotlib.pyplot as plt

# Vẽ biểu đồ phân tán giữa số giờ học mỗi ngày và điểm số học tập
fig, ax = plt.subplots()
ax.scatter(df_clean['study_hours_per_day'], df_clean['exam_score'], alpha=0.6) #Tham số alpha dùng để điều chỉnh độ trong suốt (transparency) của các điểm được vẽ.

# Ghi nhãn trục
ax.set_xlabel("Số giờ học mỗi ngày")
ax.set_ylabel("Điểm số học tập")
ax.set_title("📚 Mối liên hệ giữa số giờ học và điểm số")

# Hiển thị biểu đồ trên Streamlit
st.pyplot(fig)

from sklearn.linear_model import LinearRegression
import numpy as np

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Chuẩn bị dữ liệu
X = df_clean['study_hours_per_day']
y = df_clean['exam_score']

# Huấn luyện mô hình
model.fit(X, y)

# Dự đoán trên toàn bộ dữ liệu để tính hiệu suất
y_pred_full = model.predict(X)

# Vẽ biểu đồ phân tán
fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.6, label="Dữ liệu thực tế")

# Tạo trục x mới để vẽ đường hồi quy
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Vẽ đường hồi quy
ax.plot(x_range, y_pred, color='red', linewidth=2, label="Hồi quy tuyến tính")

# Ghi nhãn
ax.set_xlabel("Số giờ học mỗi ngày")
ax.set_ylabel("Điểm số")
ax.set_title("📈 Dự đoán điểm số từ số giờ học")
ax.legend()

# Hiển thị biểu đồ trên Streamlit
st.pyplot(fig)

# .....

from sklearn.metrics import mean_squared_error, r2_score

# Tính toán MSE và R² Score
mse = mean_squared_error(y, y_pred_full)
r2 = r2_score(y, y_pred_full)

# Hiển thị kết quả trên Streamlit
st.write("📊 Hiệu suất mô hình hồi quy tuyến tính:")
st.write(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
st.write(f"🔹 R² Score: {r2:.2f}")

# Tính hệ số tương quan Pearson giữa số giờ học mỗi ngày và điểm số
correlation = df_clean['study_hours_per_day'].corr(df_clean['exam_score'])

# Hiển thị kết quả tương quan trên Streamlit
st.write(f"🔍 Hệ số tương quan giữa số giờ học mỗi ngày và điểm số: {correlation:.2f}")

# Giải thích nhanh về hệ số tương quan
if correlation > 0.5:
    st.write("➡️ Mối tương quan mạnh và tích cực: số giờ học càng nhiều, điểm số càng cao.")
elif correlation > 0:
    st.write("➡️ Mối tương quan yếu và tích cực.")
else:
    st.write("➡️ Mối tương quan không đáng kể hoặc tiêu cực.")

#Câu 2
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

st.markdown("## 💤 Câu 2: Giấc ngủ ảnh hưởng đến điểm số thế nào?")

# Heatmap tương quan giữa các biến liên tục
numeric_cols = ['study_hours_per_day', 'sleep_hours', 'social_media_hours', 'exam_score']
corr = df_clean[numeric_cols].corr()

fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
ax1.set_title("🔍 Ma trận tương quan giữa các biến")
st.pyplot(fig1)

# Biểu đồ phân tán riêng giữa Sleep Hours và Grades
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_clean, x='sleep_hours', y='exam_score', hue='study_hours_per_day', palette='viridis', ax=ax2)
ax2.set_title("💤 Giấc ngủ vs Điểm số (tô màu theo giờ học)")
st.pyplot(fig2)

st.markdown("## ⏳ Câu 3 (mới): Số giờ học và điểm số – Có thực sự tỷ lệ thuận?")

# Nhóm số giờ học thành các nhóm rời rạc dễ nhìn
df_clean['Study Hours Group'] = pd.cut(df_clean['study_hours_per_day'], bins=[0, 1, 2, 3, 4, 5, 10], 
                                        labels=['0-1h', '1-2h', '2-3h', '3-4h', '4-5h', '5h+'])

# Điểm trung bình theo từng nhóm
mean_by_group = df_clean.groupby('Study Hours Group')['exam_score'].mean().reset_index()
st.write("📊 Điểm trung bình theo nhóm số giờ học:")
st.dataframe(mean_by_group)

# Boxplot để xem phân phối điểm số theo từng nhóm học
fig6, ax6 = plt.subplots()
sns.boxplot(data=df_clean, x='Study Hours Group', y='exam_score', palette='pastel', ax=ax6)
ax6.set_title("📦 Phân phối điểm số theo nhóm giờ học")
st.pyplot(fig6)

st.markdown("## 📱 Câu 4: Mạng xã hội ảnh hưởng thế nào đến kết quả học?")

# Biểu đồ scatter + size theo Study Hours
fig4, ax4 = plt.subplots()
sns.scatterplot(
    data=df_clean,
    x='social_media_hours',
    y='exam_score',
    size='study_hours_per_day',
    hue='sleep_hours',
    palette='cool',
    ax=ax4
)
ax4.set_title("📱 Thời gian dùng mạng xã hội vs Điểm số")
st.pyplot(fig4)

# Vẽ biểu đồ histogram của điểm số theo mức sử dụng mạng xã hội
df_clean['Social Category'] = pd.cut(df_clean['social_media_hours'], bins=[0, 2, 4, 6, 8, 12], labels=['0-2h', '2-4h', '4-6h', '6-8h', '8-12h'])

fig5, ax5 = plt.subplots()
sns.boxplot(data=df_clean, x='Social Category', y='exam_score', palette='YlGnBu', ax=ax5)
ax5.set_title("🧊 Boxplot: Điểm số theo nhóm dùng mạng xã hội")
st.pyplot(fig5)

from sklearn.linear_model import LinearRegression

st.markdown("## 📈 Phân tích hồi quy: Giấc ngủ & Mạng xã hội")

# Biến 1: Sleep Hours → exam_score
X_sleep = df_clean[['sleep_hours']]
y_score = df_clean['exam_score']

model_sleep = LinearRegression()
model_sleep.fit(X_sleep, y_score)
r2_sleep = model_sleep.score(X_sleep, y_score)

# Biến 2: Social Media Hours → exam_score
X_social = df_clean[['social_media_hours']]
model_social = LinearRegression()
model_social.fit(X_social, y_score)
r2_social = model_social.score(X_social, y_score)

# Hệ số hồi quy
coef_sleep = model_sleep.coef_[0]
coef_social = model_social.coef_[0]

# Hiển thị kết luận
st.subheader("📌 Kết luận từ mô hình hồi quy:")

if coef_sleep > 0:
    st.success(f"✔️ Mỗi giờ ngủ thêm giúp tăng điểm trung bình khoảng {coef_sleep:.2f} điểm.")
else:
    st.warning(f"⚠️ Ngủ nhiều hơn có thể liên quan đến việc giảm điểm, với hệ số {coef_sleep:.2f}.")

if coef_social < 0:
    st.success(f"✔️ Mỗi giờ dùng mạng xã hội thêm có thể làm giảm điểm trung bình khoảng {-coef_social:.2f} điểm.")
else:
    st.warning(f"⚠️ Bất ngờ! Dùng mạng xã hội nhiều hơn lại liên quan đến điểm tăng ({coef_social:.2f}).")

st.markdown("## 🔄 Tương tác: Chọn biến để so sánh với điểm số")



st.markdown("## 🍽️ Câu 5: Chế độ ăn ảnh hưởng tới sức khỏe tinh thần?")

# Đếm số học sinh theo từng kết hợp chế độ ăn và sức khỏe tâm thần
diet_vs_mental = df_clean.groupby(['diet_quality', 'mental_health_rating']).size().unstack(fill_value=0)

# Vẽ stacked bar chart
fig8, ax8 = plt.subplots()
diet_vs_mental.plot(kind='bar', stacked=True, colormap='viridis', ax=ax8)
ax8.set_title("🧠 Sức khỏe tinh thần theo chất lượng chế độ ăn")
ax8.set_ylabel("Số học sinh")
st.pyplot(fig8)

st.markdown("## 🎓 Câu 6: Học vấn của phụ huynh và hoạt động ngoại khóa")

# Lọc học sinh tham gia hoạt động ngoại khóa
extracurricular_counts = df_clean[df_clean['extracurricular_participation'] == "Yes"]['parental_education_level'].value_counts()

# Pie chart
fig9, ax9 = plt.subplots()
ax9.pie(extracurricular_counts, labels=extracurricular_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
ax9.set_title("🥧 Tỷ lệ học sinh tham gia ngoại khóa theo học vấn phụ huynh")
st.pyplot(fig9)

st.markdown("## 🏃‍♀️ Câu 7: Giới tính, tập thể dục và internet")

# Group theo giới tính và internet quality
grouped = df_clean.groupby(['gender', 'internet_quality']).size().reset_index(name='count')
st.write(grouped)

# Vẽ grouped bar plot
fig10, ax10 = plt.subplots()
sns.barplot(data=grouped, x='internet_quality', y='count', hue='gender', palette='Set1', ax=ax10)
ax10.set_title("🌐 Chất lượng internet theo giới tính")
ax10.set_ylabel("Số học sinh")
st.pyplot(fig10)
