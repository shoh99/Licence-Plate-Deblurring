from dataclasses import field
from turtle import width
import streamlit as st
from PIL import Image



image = Image.open("./logo.png") #Brand logo image (optional)

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.2])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
    
with col2:               # To display brand logo
    st.image(image,  width=150)


    #Add a header and expander in side bar
st.sidebar.markdown('<p class="font">Naver Team</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
       Naver Team project interface  \n  \nThis app was created by Naver Team
     """)


#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

# Add 'before' and 'after' columns

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)