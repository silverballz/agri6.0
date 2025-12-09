import streamlit as st

st.title("🌱 AgriFlux Test")
st.write("If you can see this, Streamlit is working!")
st.success("Dashboard is functional")

if st.button("Click me"):
    st.balloons()
    st.write("Button clicked!")
