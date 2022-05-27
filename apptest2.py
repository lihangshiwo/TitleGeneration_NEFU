import streamlit as st
def main():
    st.set_page_config(
        page_title="智能创作平台",
        page_icon=":rainbow:",
        initial_sidebar_state="auto",
        layout="wide"
    )
    placeholder = st.sidebar.empty()
    with placeholder.container():
        st.write("This is one element")
        st.write("This is another")
    placeholder.empty()
    st.sidebar.subheader("登录区域2")

if __name__=="__main__":
    main()