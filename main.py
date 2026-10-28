import streamlit as st
st.title('나의 첫 웹 서비스')
a=st.text_input('이름입력')
b=st.selectbox('좋아하는 음식 선택',['비빔밥','불고기','떡볶이'])
if st.button('인삿말 생성'):
  st.write('반갑')
