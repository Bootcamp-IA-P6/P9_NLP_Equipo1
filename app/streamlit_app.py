import streamlit as st 
import pandas as pd
import json
import sys
from pathlib import Path

# Prepara el path para poder acceder a los scripts y funciones en carpeta src
sys.path.append(".")

from src.hatescan.scraping.youtube_scraper import fetch_comments
# Import the utility to extract the ID from full URLs
from src.hatescan.utils.youtube_utils import extract_video_id


from supabase import create_client, Client
# ---------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------
# Leer secretos
SUPABASE_URL = st.secrets.connections.supabase.SUPABASE_URL2
SUPABASE_KEY = st.secrets.connections.supabase.SUPABASE_KEY2

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

# -------------------------
# Datos de prueba
# -------------------------

# OJO con los datos recibidos si son true, false o null ¿Darán problemas? ¿Python debería hacerlo sólo . Usar json_load?
# en formato json para python y streamlit como poner un valor booleano
# en python como acceder a un campo de json

prueba = '{"nombre": "Juanma", "casado": true}'

result = """{ 
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "model_used": "transformer_roberta",
    "total_comments": 150,
    "toxic_count": 23,
    "non_toxic_count": 127
}"""

resultado = """{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "model_used": "transformer_roberta",
  "total_comments": 150,
  "toxic_count": 23,
  "non_toxic_count": 127,
  "comments": [
    {
      "comment_id": "abc123",
      "text_original": "you are so stupid and pathetic",
      "is_toxic": true,
      "confidence": 0.94,
      "categories": {
        "is_hatespeech": null,
        "is_racist": null,
        "is_threat": null,
        "is_obscene": null
      }
    },
    {
      "comment_id": "def456",
      "text_original": "great video, loved it!",
      "is_toxic": false,
      "confidence": 0.97,
      "categories": {
        "is_hatespeech": null,
        "is_racist": null,
        "is_threat": null,
        "is_obscene": null
      }
    }
  ]
}"""

# -------------------------
# Funciones
# -------------------------

def login_screen():
    st.header(":orange[HateScan]")

    # Línea divisoria naranja con CSS en st.markdown
    st.markdown("<hr style='border: 2px solid orange;'>", unsafe_allow_html=True)

    with st.container(border=True, key="contenedor_login"):
        st.header("This app is private.")
        st.subheader("Please log in.")
        st.button("Log in with Google", on_click=st.login)

    # Línea divisoria naranja con CSS en st.markdown
    st.markdown("<hr style='border: 2px solid orange;'>", unsafe_allow_html=True)

def get_comments(url):
    # Clean the URL to obtain the 11-character video ID
    video_id = extract_video_id(url)
    
    if not video_id:
        print(f"Skipping: Could not extract a valid video ID from {url}")
        return []
        
    print(f"Extracted Video ID: {video_id}")
    
    # Fetch the comments using the cleaned ID
    return fetch_comments(video_id, max_results=5)

def get_users():
    # SELECT * FROM users
    response = (
        supabase
        .table("users")
        #.table("users_test")
        .select("*")
        .execute()
    )

    # Datos JSON devueltos
    return response.data

def search_user(user):
    # SELECT * FROM users
    response = (
        supabase
        .table("users")
        #.table("users_test")
        .select("*")
        .eq("email", user)
        .execute()
    )

    # Datos JSON devueltos
    return response.data

def save_comments(datos):
    # # 1. Datos externos simulados como cadena de texto
    # cadena_externa = '{"usuario": "Ana", "edad": 28, "activo": true}'

    # # 2. Convertir la cadena a un diccionario de Python
    # datos_json = json.loads(cadena_externa)

    # print(datos_json["usuario"]) # Imprime: Ana

    comment = {}
    comment["model_used"] = datos["model_used"]
    comment["text_original"] = datos["comments"][0]["text_original"] 
    comment["is_toxic"] = datos["comments"][0]["is_toxic"] 

    return comment


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
        st.write("🤓")

    # -----------------------------------------------PESTAÑAS----------------------------------------------------
    # Creamos las 3 pestañas
    tab1, tab2, tab3 = st.tabs(["URL Video", "Comments", "Pestaña 3"])

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

                        # # Clean the URL to obtain the 11-character video ID
                        # video_id = extract_video_id(url_ingresada)
                        
                        # if not video_id:
                        #     print(f"Skipping: Could not extract a valid video ID from {url_ingresada}")
                        #     # continue
                            
                        # print(f"Extracted Video ID: {video_id}")
                        
                        # # Fetch the comments using the cleaned ID
                        # comentarios = fetch_comments(video_id, max_results=5)

                        comentarios = get_comments(url_ingresada)

                    # Mostrar resultado
                    st.success("Comentarios obtenidos correctamente")
                    # st.write(comentarios)

                    # Código para probar el JSON
                    # st.write(resultado)
                    # Convertimos la cadena JSON a un diccionario de Python
                    datos = json.loads(resultado)
                    st.write(datos)
                    st.write(datos["model_used"])
                    # st.write(datos["comments"])
                    st.write(datos["comments"][0]["text_original"])

                    # Poner comentarios en formato dataframe
                    df_comments = pd.DataFrame(datos["comments"])
                    # Mostrar tabla
                    st.dataframe(df_comments)

                    # Recorrer los comentarios crear Json y actualizar Supabase con función
                    st.write(save_comments(datos))
                   
                    # Ejemplo: Mostrar la URL en un botón de enlace
                    st.link_button("Ir al video", st.session_state['ultima_url'])
                else:
                    st.warning("Por favor, introduce una URL válida primero.")

        # Raya o separador al final de la pestaña
        st.divider()
        
    with tab2:
        st.header("Datos del segundo CSV")

        boton_supabase = st.button("Supabase")
        # if st.button("Supabase"):
        if boton_supabase:

            # --------------------------------------------------------
            # Ejecutamos el codigo python para conectar con supabase
            # --------------------------------------------------------
            try:
                # Spinner mientras consulta
                with st.spinner("Consultando usuarios en Supabase..."):

                    # df_users = pd.DataFrame(get_users())
                    # df_users = pd.DataFrame(search_user("juanmanuel.iriondo@gmail.com"))
                    df_users = pd.DataFrame(search_user("juana@gmail.com"))
                    # df_users = pd.DataFrame(search_user(st.user.email))

                    # Comprobamos que ha devuelto algo
                    if df_users.empty:
                        st.write("El usuario No existe")
                    else:
                        st.write("El usuario YA existe")
                        st.write(df_users["id"][0])

                        dato = df_users["id"][0] + 1
                        st.write(dato)

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

    st.button("Log out", on_click=st.logout)