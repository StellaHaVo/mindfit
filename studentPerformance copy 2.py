import kagglehub

# Download latest version
path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")

import streamlit as st

st.markdown(
    """
    <h2 style='text-align: center;'>
        Dự án cuối khóa CSA15
    </h2>
    <h1 style='text-align: center;'>
        Phân tích các thói quen của học sinh ảnh hưởng đến kết quả học tập
    </h1>
    <h2 style='text-align: center;'>
        Võ Ngọc Thiên Hà
    </h2>
    """,
    unsafe_allow_html=True
)


st.subheader("Tiêu đề 2")

import streamlit as st
import pandas as pd

st.title("I. Tìm hiểu dữ liệu")

# Tải dữ liệu từ file CSV 
df = pd.read_csv(r"CSA\L10\CP03\student_data.csv")  \
    
# Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu
st.write("📊 Dữ liệu ban đầu:", df.head())

df.columns = ['ID','Tuổi','Giới tính','Thời gian học/ngày (giờ)','Thời gian dùng MXH (giờ)','Thời gian Netflix (giờ)','Việc làm thêm (Y/N)','Phần trăm buổi học (%)','Thời gian ngủ (giờ)','Chất lượng bữa ăn','Mức độ thường xuyên tập thể dục','Trình độ học vấn của ba mẹ','Chất lượng Internet','Đánh giá tinh thần','Tham gia HĐNK','Điểm số']

df.set_index("ID", inplace=True)

# Kiểm tra xem có giá trị nào bị thiếu không
df_nulls = df.isnull().sum().to_frame(name="Số giá trị thiếu")
st.write("📉 Tổng số giá trị thiếu theo cột:")

st.write(df_nulls)

# Xóa các dòng có giá trị thiếu để đảm bảo phân tích chính xác
df_clean = df.dropna()

# Hiển thị lại dữ liệu sau khi làm sạch
st.write("✅ Dữ liệu sau khi làm sạch:", df_clean.head())

import matplotlib.pyplot as plt

st.title("II. Phân tích dữ liệu")
st.header("Phân tích các mối liên hệ với điểm số")

# Dropdown chọn biến để phân tích
selected_var = st.selectbox("📌 Chọn biến để xem mối liên hệ với điểm số:",
                            ['Thời gian học/ngày (giờ)', 'Thời gian ngủ (giờ)', 'Thời gian dùng MXH (giờ)','Thời gian Netflix (giờ)','Phần trăm buổi học (%)','Đánh giá tinh thần','Mức độ thường xuyên tập thể dục'])

# Chuẩn bị dữ liệu cho mô hình
X = df_clean[[selected_var]]  # giữ dạng DataFrame 2D
y = df_clean['Điểm số']

# Huấn luyện mô hình hồi quy
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()
model.fit(X, y)

# Tạo giá trị X mịn để vẽ đường hồi quy
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

# Vẽ biểu đồ kết hợp dữ liệu và đường hồi quy
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(X, y, alpha=0.6, label="Dữ liệu thực tế")
ax.plot(x_range, y_pred, color='red', linewidth=2, label="Hồi quy tuyến tính")

# Ghi nhãn
ax.set_xlabel(selected_var)
ax.set_ylabel("Điểm số")
ax.set_title(f"📈 Dự đoán điểm số từ {selected_var}")
ax.set_ylim(0, 100)  # Giới hạn trục y từ 0 đến 100
ax.legend(loc="lower right")

# Hiển thị trên Streamlit
st.pyplot(fig)

from sklearn.metrics import mean_squared_error, r2_score

# Dự đoán trên toàn bộ dữ liệu để tính hiệu suất
y_pred_full = model.predict(X)

# Tính toán MSE và R² Score
mse = mean_squared_error(y, y_pred_full)
r2 = r2_score(y, y_pred_full)

# Hiển thị kết quả trên Streamlit
st.write("📊 Hiệu suất mô hình hồi quy tuyến tính:")
st.write(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
st.write(f"🔹 R² Score: {r2:.2f}")

# Tính hệ số tương quan Pearson giữa số giờ học mỗi ngày và điểm số
correlation = df_clean['Thời gian học/ngày (giờ)'].corr(df_clean['Điểm số'])

# Hiển thị kết quả tương quan trên Streamlit
st.write(f"🔍 Hệ số tương quan giữa {selected_var} và điểm số: {correlation:.2f}")

# Giải thích nhanh về hệ số tương quan
if correlation > 0.5:
    st.write(f"➡️ Mối tương quan mạnh và tích cực: {selected_var} càng nhiều, điểm số càng cao.")
elif correlation > 0:
    st.write("➡️ Mối tương quan yếu và tích cực.")
else:
    st.write("➡️ Mối tương quan không đáng kể hoặc tiêu cực.")


import seaborn as sns

# Heatmap tương quan giữa các biến liên tục
numeric_cols = ['Thời gian học/ngày (giờ)', 'Thời gian ngủ (giờ)', 'Thời gian dùng MXH (giờ)', 'Điểm số']
corr = df_clean[numeric_cols].corr()

fig1, ax1 = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax1)
ax1.set_title("🔍 Ma trận tương quan giữa các biến")
st.pyplot(fig1)

# Biểu đồ phân tán riêng giữa Sleep Hours và Grades
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_clean, x='Thời gian dùng MXH (giờ)', y='Thời gian Netflix (giờ)', hue='Thời gian học/ngày (giờ)', palette='viridis', ax=ax2)
ax2.set_title("Thời gian MXH vs Netflix (tô màu theo giờ học)")
st.pyplot(fig2)

st.markdown("Số giờ học và điểm số – Có thực sự tỷ lệ thuận?")

# Nhóm số giờ học thành các nhóm rời rạc dễ nhìn
df_clean['Study Hours Group'] = pd.cut(df_clean['Thời gian học/ngày (giờ)'], bins=[0, 1, 2, 3, 4, 5, 10], 
                                        labels=['0-1h', '1-2h', '2-3h', '3-4h', '4-5h', '5h+'])

# Điểm trung bình theo từng nhóm
mean_by_group = df_clean.groupby('Study Hours Group')['Điểm số'].mean().reset_index()
st.write("📊 Điểm trung bình theo nhóm số giờ học:")
st.dataframe(mean_by_group)

# Boxplot để xem phân phối điểm số theo từng nhóm học
fig3, ax3 = plt.subplots()
sns.boxplot(data=df_clean, x='Study Hours Group', y='Điểm số', palette='pastel', ax=ax3)
ax3.set_title("📦 Phân phối điểm số theo nhóm giờ học")
st.pyplot(fig3)

st.markdown("Chế độ ăn ảnh hưởng tới sức khỏe tinh thần?")

# Đếm số học sinh theo từng kết hợp chế độ ăn và sức khỏe tâm thần
diet_vs_mental = df_clean.groupby(['Chất lượng bữa ăn', 'Đánh giá tinh thần']).size().unstack(fill_value=0) #điền giá trị thiếu vào nếu không có

# Vẽ stacked bar chart
fig8, ax8 = plt.subplots()
diet_vs_mental.plot(kind='bar', stacked=True, colormap='viridis', ax=ax8)
ax8.set_title("🧠 Sức khỏe tinh thần theo chất lượng chế độ ăn")
ax8.set_ylabel("Số học sinh")
st.pyplot(fig8)

# Lọc học sinh có điểm số > 80
high_score_counts = df_clean[df_clean['Điểm số'] > 80]['Trình độ học vấn của ba mẹ'].value_counts()

# Pie chart
fig9, ax9 = plt.subplots()
ax9.pie(high_score_counts, labels=high_score_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
ax9.set_title("🥧 Tỷ lệ học sinh có điểm > 80 theo học vấn phụ huynh")
st.pyplot(fig9)

st.markdown("## 📈 Phân tích hồi quy: Trình độ học vấn của phụ huynh & Điểm số")

# Chuyển đổi Trình độ học vấn của ba mẹ thành dạng số
df_clean['parental_education_numeric'] = df_clean['Trình độ học vấn của ba mẹ'].astype('category').cat.codes

# Biến độc lập (X) và phụ thuộc (y)
X_parent_edu = df_clean[['parental_education_numeric']]
y_score = df_clean['Điểm số']

# Tạo mô hình hồi quy
model_parent_edu = LinearRegression()
model_parent_edu.fit(X_parent_edu, y_score)
r2_parent_edu = model_parent_edu.score(X_parent_edu, y_score)

# Hệ số hồi quy
coef_parent_edu = model_parent_edu.coef_[0]

# Hiển thị kết luận
st.subheader("📌 Kết luận từ mô hình hồi quy:")

if coef_parent_edu > 0:
    st.success(f"✔️ Mỗi mức học vấn cao hơn của phụ huynh có thể giúp tăng điểm trung bình khoảng {coef_parent_edu:.2f} điểm.")
else:
    st.warning(f"⚠️ Ngạc nhiên! Trình độ học vấn cao hơn của phụ huynh lại có liên quan đến giảm điểm ({coef_parent_edu:.2f}).")

# Ép cột Chất lượng bữa ăn về dạng phân loại có thứ tự rõ ràng
diet_order = ['Good', 'Fair', 'Poor']
df_clean['Chất lượng bữa ăn'] = pd.Categorical(df_clean['Chất lượng bữa ăn'], categories=diet_order, ordered=True)

# Tính trung bình sức khỏe tinh thần
summary = df_clean.groupby(['Chất lượng bữa ăn', 'Mức độ thường xuyên tập thể dục'])['Đánh giá tinh thần'].mean().reset_index()

# Hiển thị bảng trên Streamlit
st.write("🧠 Trung bình sức khỏe tinh thần theo chế độ ăn (theo thứ tự Poor → Good) và tập thể dục:")
st.dataframe(summary)

# Pivot lại để vẽ heatmap đúng trục
heatmap_data = summary.pivot(index='Chất lượng bữa ăn', columns='Mức độ thường xuyên tập thể dục', values='Đánh giá tinh thần')

# Vẽ heatmap
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=ax)
ax.set_title("🌿 Sức khỏe tinh thần theo Ăn uống và Tập thể dục")
st.pyplot(fig)


# “Học sinh ăn uống tốt và tập đều đặn có tinh thần tốt hơn không?”

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("## 🏃‍♀️ Câu 7: Tỷ lệ điểm danh, tập thể dục và internet")

# Phân loại Phần trăm buổi học (%) thành nhóm (ví dụ: thấp, trung bình, cao)
df_clean['attendance_group'] = pd.cut(df_clean['Phần trăm buổi học (%)'], bins=[0, 50, 80, 100], labels=["Thấp", "Trung Bình", "Cao"])

# Nhóm dữ liệu theo attendance_group và Chất lượng Internet
grouped = df_clean.groupby(['attendance_group', 'Chất lượng Internet']).size().reset_index(name='count')
st.write(grouped)

# Vẽ grouped bar plot
fig10, ax10 = plt.subplots()
sns.barplot(data=grouped, x='Chất lượng Internet', y='count', hue='attendance_group', palette='Set1', ax=ax10)
ax10.set_title("🌐 Chất lượng internet theo nhóm điểm danh")
ax10.set_ylabel("Số học sinh")
st.pyplot(fig10)

st.markdown("## ⚙️ Tuỳ chỉnh điều kiện Lối sống cân bằng")

# --- INPUT ĐIỀU KIỆN NGỦ ---
min_sleep = st.slider("📏 Ngủ ít nhất (giờ)", 4.0, 10.0, 7.0, 0.5)
max_sleep = st.slider("📏 Ngủ nhiều nhất (giờ)", 5.0, 12.0, 9.0, 0.5)

# --- INPUT ĐIỀU KIỆN HỌC ---
min_study = st.slider("📚 Học ít nhất (giờ/ngày)", 0.0, 8.0, 1.0, 0.5)
max_study = st.slider("📚 Học nhiều nhất (giờ/ngày)", 0.0, 10.0, 4.0, 0.5)

# --- INPUT ĐIỀU KIỆN TẬP ---
min_exercise = st.number_input("🏃‍♀️ Số lần tập thể dục tối thiểu mỗi tuần", min_value=0, max_value=14, value=3)

# Tạo cột 'balanced_lifestyle' theo điều kiện người dùng chọn
df_clean['balanced_lifestyle'] = (
    (df_clean['Thời gian ngủ (giờ)'] >= min_sleep) & (df_clean['Thời gian ngủ (giờ)'] <= max_sleep) &
    (df_clean['Thời gian học/ngày (giờ)'] >= min_study) & (df_clean['Thời gian học/ngày (giờ)'] <= max_study) &
    (df_clean['Mức độ thường xuyên tập thể dục'] >= min_exercise)
)

# Chuyển True/False thành chuỗi
df_clean['balanced_lifestyle'] = df_clean['balanced_lifestyle'].map({True: "Balanced", False: "Unbalanced"})

# Tổng hợp dữ liệu
summary = df_clean.groupby('balanced_lifestyle')[['Điểm số', 'Đánh giá tinh thần']].mean().reset_index()
summary = summary.sort_values(by='Điểm số', ascending=False)

# Hiển thị bảng
st.write("📊 Trung bình điểm số và sức khỏe tinh thần theo lối sống bạn định nghĩa:")
st.dataframe(summary)

# Vẽ biểu đồ bar
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
melted = summary.melt(id_vars='balanced_lifestyle', var_name='Chỉ số', value_name='Giá trị')

sns.barplot(data=melted, x='balanced_lifestyle', y='Giá trị', hue='Chỉ số', palette='coolwarm', ax=ax)
ax.set_title("🎯 So sánh lối sống theo tuỳ chọn của bạn")
st.pyplot(fig)
