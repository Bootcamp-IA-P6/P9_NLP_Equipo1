import streamlit as st 
import pandas as pd

import sys
# import os
from pathlib import Path

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(".")

from src.hatescan.scraping.youtube_scraper import fetch_comments
# Import the utility to extract the ID from full URLs
from src.hatescan.utils.youtube_utils import extract_video_id

# sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
# from hatescan.scraping.youtube_scraper import fetch_comments
# # Import the utility to extract the ID from full URLs
# from hatescan.utils.youtube_utils import extract_video_id


from supabase import create_client, Client
# ---------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------
# Leer secretos
#SUPABASE_URL = st.secrets.connections.supabase.SUPABASE_URL
#SUPABASE_KEY = st.secrets.connections.supabase.SUPABASE_KEY

SUPABASE_URL = st.secrets.connections.supabase.SUPABASE_URL3
SUPABASE_KEY = st.secrets.connections.supabase.SUPABASE_KEY3

# Crear conexión
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Fin Supabase ---------------------------------------------------------



# Definimos el estilo CSS apuntando al 'key' del contenedor
css = """
.st-key-contenedor_login {
    background-color: rgba(173, 216, 230, 0.5); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}
.st-key-contenedor_personalizado {
    background-color: rgba(173, 216, 230, 0.3); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}
.st-key-contenedor_url {
    background-color: rgba(173, 216, 230, 0.1); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}
"""

def login_screen():
    st.header(":orange[HateScan]")
    # st.header("Este es un header :blue[azul] y :red[rojo]")

    # Línea divisoria naranja con CSS en st.markdown
    st.markdown("<hr style='border: 2px solid orange;'>", unsafe_allow_html=True)

    with st.container(border=True, key="contenedor_login"):
        st.header("This app is private.")
        st.subheader("Please log in.")
        st.button("Log in with Google", on_click=st.login)

    # Línea divisoria naranja con CSS en st.markdown
    st.markdown("<hr style='border: 2px solid orange;'>", unsafe_allow_html=True)

def get_users():
    # SELECT * FROM users
    response = (
        supabase
        #.table("users")
        .table("users_test")
        .select("*")
        .eq("email", "juanmanuel.iriondo@gmail.com")
        .execute()
    )

    # Datos JSON devueltos
    return response.data

def search_user(user):
    # SELECT * FROM users
    response = (
        supabase
        #.table("users")
        .table("users_test")
        .select("*")
        .eq("email", user)
        .execute()
    )

    # Datos JSON devueltos
    return response.data

# -------------------------
# Programa
# -------------------------

# Inyectamos el CSS en la aplicación
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if not st.user.is_logged_in:
    login_screen()
else:
    # Usando contenedores con borde
    with st.container(border=True, key="contenedor_personalizado"):
        st.header(f"Welcome, {st.user.name}!")
        st.write(f"**Correo electrónico:** \n{st.user.email}") 


    # -----------------------------------------------PESTAÑAS----------------------------------------------------
    # Creamos las 3 pestañas
    tab1, tab2, tab3 = st.tabs(["Pestaña 1", "Pestaña 2", "Pestaña 3"])

    df1 = pd.read_csv("data/raw/comments_dQw4w9WgXcQ_20260511.csv")

    with tab1:
        st.header("Datos del primer CSV")
            # Usando contenedores con borde
        with st.container(border=True, key="contenedor_url"):

            # 1. Pedir al usuario que ingrese la URL
            url_ingresada = st.text_input("Introduce una URL de un video de Youtube (ej. https://google.com):")

            # 2. Botón de acción
            if st.button("Procesar URL"):
                if url_ingresada:
                    # Guardamos en sesión para mantener el estado
                    st.session_state['ultima_url'] = url_ingresada
                    
                    # Aquí ejecutas lo que necesites hacer con la URL
                    st.success(f"¡URL recibida con éxito!")
                    st.write(f"Accediendo a: {st.session_state['ultima_url']}")

                    # ------------------------------------------
                    # Ejecutamos el codigo python
                    # ------------------------------------------
                    # Ejecutar scraping
                    with st.spinner("Extrayendo comentarios de YouTube..."):
                        # comentarios = fetch_comments(url_ingresada, max_results=5)

                        # Clean the URL to obtain the 11-character video ID
                        video_id = extract_video_id(url_ingresada)
                        
                        if not video_id:
                            print(f"Skipping: Could not extract a valid video ID from {url_ingresada}")
                            # continue
                            
                        print(f"Extracted Video ID: {video_id}")
                        
                        # Fetch the comments using the cleaned ID
                        comentarios = fetch_comments(video_id, max_results=5)

                    # Mostrar resultado
                    st.success("Comentarios obtenidos correctamente")
                    st.write(comentarios)
                    
                    # Ejemplo: Mostrar la URL en un botón de enlace
                    st.link_button("Ir al sitio", st.session_state['ultima_url'])
                else:
                    st.warning("Por favor, introduce una URL válida primero.")

        # Raya o separador al final de la pestaña
        st.divider()
        
    with tab2:
        st.header("Datos del segundo CSV")

        boton_supabase = st.button("Supabase")
        # if st.button("Supabase"):
        if boton_supabase:
            st.write("boton Supabase pulsado")
            st.success("¡Hola! Has pulsado el botón correctamente.")

            # --------------------------------------------------------
            # Ejecutamos el codigo python para conectar con supabase
            # --------------------------------------------------------
            try:

                # Spinner mientras consulta
                with st.spinner("Consultando usuarios en Supabase..."):

                    # # SELECT * FROM users
                    # response = (
                    #     supabase
                    #     #.table("users")
                    #     .table("users_test")
                    #     .select("*")
                    #     .eq("email", "juanmanuel.iriondo@gmail.com")
                    #     .execute()
                    # )

                    # # Datos JSON devueltos
                    # usuarios = response.data

                    # Convertimos a DataFrame
                    # df_users = pd.DataFrame(usuarios)

                    # df_users = pd.DataFrame(get_users())
                    df_users = pd.DataFrame(search_user("juanmanuel.iriondo@gmail.es"))

                    # Comprobamos que ha devuelto algo
                    if df_users.empty:
                        st.write("El usuario No existe")
                    else:
                        st.write("El usuario YA existe")


                st.success("Usuarios obtenidos correctamente")

                # Mostrar tabla
                st.dataframe(df_users)

            except Exception as e:

                st.error(f"Error conectando con Supabase: {e}")

            # --------------------FIN conectar supabase-------------------------------------

        # Raya o separador al final de la pestaña
        st.divider()
        
    with tab3:
        st.header("Datos del tercer CSV")
        st.dataframe(df1) # Muestra la tabla interactiva

        # Raya o separador al final de la pestaña
        st.divider()
    # -----------------------------------------------FIN DE PESTAÑAS----------------------------------------------


    # Colorear textos específicos de forma nativa
    st.markdown(":blue-background[Este texto tiene un fondo azul]")
    st.markdown(":orange[Este texto es de color naranja]")

    st.button("Log out", on_click=st.logout)
