import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Hello bd")
st.write("Es ist meine Erste Mal die Web zu benutzen")
st.header("warum kann ich nicht diese App benutzen")
st.subheader("wassss ich vermisse euch sehr sehr vielpyh")
data = {
    'course': ['web deve','compusci'],
    'age': ['13-17','18-20']
}

st.header("Hiển thị biểu đồ trên web application bằng streamlit")
x = np.linspace(0,10,100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x,y)
st.pyplot(fig)

