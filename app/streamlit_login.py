import streamlit as st

def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)


# -------------------------
# Programa
# -------------------------

if not st.user.is_logged_in:
    login_screen()
else:
    # st.user
    st.header(f"Welcome, {st.user.name}!")


st.button("Log out", on_click=st.logout)