import streamlit as st 
import requests
import pandas as pd
import json
import sys
from pathlib import Path
import ast

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Prepara el path para poder acceder a los scripts y funciones en carpeta src
sys.path.append(".")

from src.hatescan.scraping.youtube_scraper import fetch_comments
# Import the utility to extract the ID from full URLs
from src.hatescan.utils.youtube_utils import extract_video_id

from src.hatescan.database.database import save_hatescan_results

# Importaciones para probar el modelo
from src.hatescan.models.predictor import HateScanPredictor



from supabase import create_client, Client
# ---------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------
# Leer secretos
SUPABASE_URL = st.secrets.connections.supabase.SUPABASE_URL
SUPABASE_KEY = st.secrets.connections.supabase.SUPABASE_KEY

# Crear conexión
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Fin Supabase ---------------------------------------------------------


# Definimos el estilo CSS apuntando al 'key' del contenedor
css = """
.st-key-contenedor_login {
    background-color: rgba(138, 218, 242, 0.5); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}

.st-key-contenedor_personalizado {
    background-color: rgba(138, 218, 242, 0.3); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}

.st-key-contenedor_url {
    background-color: rgba(138, 218, 242, 0.1); /* Fondo azul claro */
    border: 2px solid #2b5c8f;
    border-radius: 10px;
    padding: 15px;
}

/* Cambiar el fondo y el color del texto del botón */
div.stButton > button:first-child {
    background-color: #F07B3E;
    color: #ffffff;
    border-radius: 25px; /* Bordes redondeados */
    transition: background-color 1s ease !important;
}
        
/* Cambiar el color al pasar el ratón por encima (hover) */
div.stButton > button:hover {
    background-color: #00ccff;
    color: #000000;
}

/* Colorear metricas */
div[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1); /* Color de fondo con transparencia */
    border: 1px solid rgba(28, 131, 225, 0.3); /* Borde opcional */
    padding: 15px; /* Espaciado interno */
    border-radius: 10px; /* Bordes redondeados */
    color: #1E6777; /* Color del texto dentro del recuadro */
}
"""

# -------------------------
# Datos de prueba
# -------------------------

# --------------------------------------------
# Datos para probar las inserciones de datos
# --------------------------------------------

session_user = "coder_test_session" # Hardcodeado
video_title = "HateScan" # Me lo invento
# video_id = "dQw4w9WgXcQ" # Se puede sacar de la URL

resultado = {
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "model_used": "transformer_roberta",
  "total_comments": 150,
  "toxic_count": 23,
  "non_toxic_count": 127,
  "comments": [
    {
      "comment_id": "pru123",
      "text_original": "you are so stupid and pathetic",
      "is_toxic": True,
      "confidence": 0.94,
      "categories": {
        "is_hatespeech": None,
        "is_racist": None,
        "is_threat": None,
        "is_obscene": None
      }
    },
    {
      "comment_id": "pru456",
      "text_original": "great video, loved it!",
      "is_toxic": False,
      "confidence": 0.97,
      "categories": {
        "is_hatespeech": None,
        "is_racist": None,
        "is_threat": None,
        "is_obscene": None
      }
    }
  ]
}

# -------------------------
# Funciones
# -------------------------

def login_screen():

    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.header(":orange[HateScan]")
            
        with col2:
            st.header("🤬🤓")

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


def show_searches():
    # SELECT * FROM searches
    response = (
        supabase
        .table("searches")
        .select("*")
        .execute()
    )

    # Datos JSON devueltos
    return response.data


def show_comments():
    # SELECT * FROM comments
    response = (
        supabase
        .table("comments")
        .select("*")
        .execute()
    )

    # Datos JSON devueltos
    return response.data

def dashboard_models_prueba():
    st.header("DashBoard")

    df = pd.read_csv("data/raw/metricas.csv")
    st.dataframe(df) # Muestra la tabla interactiva

    # Convertir columnas en numéricas
    columnas = ['Duration_sg', 'f1_macro', 'f1_macro_test', 'f1_macro_train']
    df[columnas] = df[columnas].apply(pd.to_numeric)

    # print(df.info())
    
    # ------------------------------------------------------------
    # Botones y filtros
    # ------------------------------------------------------------

    # Separador visual opcional
    st.markdown("---")

    # Fila 1: Contenedor horizontal con multiselect
    with st.container(horizontal=True, border=True):
        st.write("Filtros")
        seleccion = st.multiselect(
            "Selecciona tu(s) modelo(s)",
             options=df['Run Name'].unique(),
             default=df['Run Name'].unique()
             )

    # Mostrar resultados seleccionados
    # st.write("Has seleccionado:", seleccion)

    # Fila 2: Contenedor horizontal con botones
    with st.container(horizontal=True, border=True):
        btn_duration = st.button("Duration")
        btn_f1_macro = st.button("F1_macro")
        btn_f1_macro_global = st.button("F1_macro_global")
        btn_f1_macro_test = st.button("F1_macro_test")

    df_filtered = df[(df['Run Name'].isin(seleccion))]

    if btn_duration:
         # Crear tu gráfico con Plotly
        fig_duration = px.line(df_filtered, x='Run Name', y='Duration_sg',
                            title='Duración en segundos',
                            labels={'Run Name': 'Modelo'})
        
        # Poner grosor de línea
        fig_duration.update_traces(line=dict(width=3))
        st.plotly_chart(fig_duration)

    if btn_f1_macro:
        fig_f1_macro = px.bar(df_filtered, x="Run Name", y="f1_macro", title="F1_macro")
        st.plotly_chart(fig_f1_macro)

    if btn_f1_macro_global:
        fig_f1_macro_global = px.bar(df_filtered, x="Run Name", y=["f1_macro", "f1_macro_test", "f1_macro_train"], title="F1_macro_global")
        st.plotly_chart(fig_f1_macro_global, use_container_width=True)

    if btn_f1_macro_test:
        df_f1_macro_test_sort = df_filtered.sort_values(by="f1_macro_test")
        fig_f1_macro_test = px.bar(df_f1_macro_test_sort, x="f1_macro_test", y="Run Name", orientation='h', title="F1_macro_test")
        st.plotly_chart(fig_f1_macro_test)


def dashboard_models():
    st.header("DashBoard")

    # df_read = pd.read_csv("data/raw/mlflow_metricas.csv")
    df_read = pd.read_csv("data/raw/mlflow_metricas (2).csv")
    st.dataframe(df_read) # Muestra la tabla interactiva

    columnas_dashboard = ['Name', 'f1_macro', 'f1_macro_test', 'f1_macro_train', 'recall', 'recall_test']
    df = df_read[columnas_dashboard]

    # Convertir columnas en numéricas
    columnas = ['f1_macro', 'f1_macro_test', 'f1_macro_train', 'recall', 'recall_test']
    df[columnas] = df[columnas].apply(pd.to_numeric)
    
    # ------------------------------------------------------------
    # Botones y filtros
    # ------------------------------------------------------------

    # Separador visual opcional
    st.markdown("---")

    # Fila 1: Contenedor horizontal con multiselect
    with st.container(horizontal=True, border=True):
        st.write("Filtros")
        seleccion = st.multiselect(
            "Selecciona tu(s) modelo(s)",
             options=df['Name'].unique(),
             default=df['Name'].unique()
             )

    # Fila 2: Contenedor horizontal con botones
    with st.container(horizontal=True, border=True):
        btn_f1_macro = st.button("F1_macro")
        btn_f1_macro_global = st.button("F1_macro_global")
        btn_recall = st.button("Recall")
        btn_recall_test = st.button("Recall_test")
        btn_recall_global = st.button("Recall_global")

    df_filtered = df[(df['Name'].isin(seleccion))]

    if btn_f1_macro:
        fig_f1_macro = px.bar(df_filtered, x="Name", y="f1_macro", title="F1_macro")
        st.plotly_chart(fig_f1_macro)

    if btn_f1_macro_global:
        fig_f1_macro_global = px.bar(df_filtered, x="Name", y=["f1_macro", "f1_macro_test", "f1_macro_train"], title="F1_macro_global")
        st.plotly_chart(fig_f1_macro_global, use_container_width=True)


    if btn_recall:
        fig_recall = px.bar(df_filtered, x="Name", y="recall", title="Recall")
        st.plotly_chart(fig_recall)

    if btn_recall_test:
        df_recall_test_sort = df_filtered.sort_values(by="recall_test")
        fig_recall_test = px.bar(df_recall_test_sort, x="recall_test", y="Name", orientation='h', title="Recall_test")
        st.plotly_chart(fig_recall_test)

    if btn_recall_global:
        fig_recall_global = px.bar(df_filtered, x="Name", y=["recall", "recall_test"], title="Recall_global")
        st.plotly_chart(fig_recall_global, use_container_width=True)


def dashboard_comments():
    # --------------------------------------------------------
    # Ejecutamos el codigo python para conectar con supabase
    # --------------------------------------------------------
    try:
        # Spinner mientras consulta
        with st.spinner("Consultando comments en Supabase..."):
            dash_comments = pd.DataFrame(show_comments())

        # Metricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Toxicos", f"{dash_comments['is_toxic'].sum()}", border=True)
        with col2:
            st.metric("No Toxicos", f"{(dash_comments['is_toxic'] == False).sum()}", border=True)
        with col3:
            st.metric("Número Comentarios", f"{len(dash_comments)}", border=True)
        with col4:
            st.metric("Media Toxicos", f"{dash_comments['is_toxic'].sum() / len(dash_comments) * 100:,.1f} %", border=True)


        conteo = dash_comments['is_toxic'].value_counts(dropna=False).reset_index()
        conteo.columns = ['Valor', 'Cantidad']

        # Gráficos principales
        col1, col2 = st.columns(2)
        with col1:
            # Crear el gráfico de barras con Plotly Express
            fig_barras = px.bar(
                conteo, 
                x='Valor', 
                y='Cantidad', 
                title='Distribución de Valores de is_toxic',
                color='Valor',
                text_auto=True # Muestra el número exacto encima de cada barra
            )
            st.plotly_chart(fig_barras)
        with col2:
            # Crear el gráfico de circulo con Plotly Express
            fig_circulo = px.pie(
                conteo, 
                names='Valor', 
                values='Cantidad', 
                title='Distribución de Valores de is_toxic',
                color='Valor',
                color_discrete_map={True: '#2CA02C', False: '#D62728'}
            )
            st.plotly_chart(fig_circulo)

    except Exception as e:
        st.error(f"Error conectando con Supabase: {e}")

    # --------------------FIN conectar supabase-------------------------------------


# -------------------------
# Programa Principal
# -------------------------

# st.set_page_config(page_title="HateScan", page_icon="🤬", layout="wide")
st.set_page_config(page_title="HateScan", page_icon="🤬")

# Inyectamos el CSS en la aplicación
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

if not st.user.is_logged_in:
    login_screen()
else:
    # Usando contenedores con borde
    with st.container(border=True, key="contenedor_personalizado"):
        col11, col12 = st.columns([4, 1])

        with col11:
            st.header(f"Welcome, {st.user.name}!")
            st.write(f"**Correo electrónico:** \n{st.user.email}")
        with col12:
            # Verificamos si el objeto de usuario contiene una URL de imagen de perfil.
            if st.user.picture:
                try:
                    # Usamos la librería 'requests' para hacer una petición GET a la URL de la imagen.
                    response = requests.get(st.user.picture)
                    # Si el código de estado de la respuesta es 200 (OK), significa que la imagen se obtuvo correctamente.
                    if response.status_code == 200:
                        # Mostramos la imagen en la aplicación. response.content contiene los bytes de la imagen.
                        st.image(response.content, width=100)
                    else:
                        # Si hay un problema al descargar la imagen (ej: error 404), mostramos una advertencia.
                        st.warning("No se pudo cargar la imagen de perfil.")
                except Exception as e:
                    # Capturamos cualquier otra excepción (ej: problemas de red) y mostramos un mensaje de error.
                    st.warning(f"Error al cargar la imagen: {e}")
            else:
                # Si el usuario de Google no tiene una imagen de perfil, informamos de ello.
                st.info("No hay imagen de perfil disponible.")

        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.header(":orange[HateScan]")
            
        with col2:
            st.header("🤬🤓")

    # -----------------------------------------------PESTAÑAS----------------------------------------------------
    # Creamos las 4 pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["URL Video", "Comments", "Searches", "DashBoard Comments", "Dashboard Models"])

    with tab1:
        st.header("Datos del primer CSV")
            # Usando contenedores con borde
        with st.container(border=True, key="contenedor_url"):

            # 1. Pedir al usuario que ingrese la URL
            url_ingresada = st.text_input("Introduce una URL de un video de Youtube (ej. https://www.youtube.com/watch?v=???????????):")

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
                        comentarios = get_comments(url_ingresada)

                    # Mostrar resultado
                    st.success("Comentarios obtenidos correctamente")
                    st.write(comentarios)

                    # ----------------------------------------------------------------------------
                    # Ejecutamos el codigo python para guardar los comments y searches en supabase
                    # ----------------------------------------------------------------------------
                    try:
                        video_id = extract_video_id(url_ingresada)
                    
                        if not video_id:
                            print(f"Skipping: Could not extract a valid video ID from {url_ingresada}")


                        # fecha y hora actual
                        now = datetime.now()
                        fecha_hora_actual = now.strftime("%Y%m%d_%H%M%S")

                        result_comments = [
                            {
                                "comment_id": f"{video_id}_{fecha_hora_actual}_c_{i}",
                                "text": item["comment"]
                            }
                            for i, item in enumerate(comentarios)
                        ]

                        # save_hatescan_results(resultado, session_user, video_title, video_id)
                        # st.success("Datos insertados correctamente")

                        predictor = HateScanPredictor()   # carga el modelo una vez al arrancar
                        result = predictor.predict(
                            video_url=url_ingresada,
                            video_id=video_id,
                            video_title="Titulo desde Streamlit",
                            comments=result_comments,
                            user_session=st.user.email,
                            save_to_db=True,
                        )
                        st.success("Datos insertados correctamente")

                    except Exception as e:
                        st.error(f"Error conectando con Supabase: {e}")
                    # --------------------FIN guardar comments y searches en supabase-------------------------------------
                   
                    # Ejemplo: Mostrar la URL en un botón de enlace
                    st.link_button("Ir al video", st.session_state['ultima_url'])
                else:
                    st.warning("Por favor, introduce una URL válida primero.")

        # Raya o separador al final de la pestaña
        st.divider()
        
    with tab2:
        st.header("Comments")

        if st.button("Show_Comments"):

            # --------------------------------------------------------
            # Ejecutamos el codigo python para conectar con supabase
            # --------------------------------------------------------
            try:
                # Spinner mientras consulta
                with st.spinner("Consultando comments en Supabase..."):
                    df_comments = pd.DataFrame(show_comments())

                st.success("Comentarios obtenidos correctamente")
                # Mostrar tabla
                st.dataframe(df_comments)

            except Exception as e:
                st.error(f"Error conectando con Supabase: {e}")

            # --------------------FIN conectar supabase-------------------------------------

        # Raya o separador al final de la pestaña
        st.divider()

    with tab3:
        st.header("Searches")

        if st.button("Show_Searches"):

            # --------------------------------------------------------
            # Ejecutamos el codigo python para conectar con supabase
            # --------------------------------------------------------
            try:
                # Spinner mientras consulta
                with st.spinner("Consultando searches en Supabase..."):
                    df_searches = pd.DataFrame(show_searches())

                st.success("Busquedas obtenidas correctamente")
                # Mostrar tabla
                st.dataframe(df_searches)

            except Exception as e:
                st.error(f"Error conectando con Supabase: {e}")

            # --------------------FIN conectar supabase-------------------------------------

        # Raya o separador al final de la pestaña
        st.divider()

    with tab4:
        dashboard_comments()

        #Raya o separador al final de la pestaña
        st.divider()
        
    with tab5:
        dashboard_models()

        # Raya o separador al final de la pestaña
        st.divider()
    # -----------------------------------------------FIN DE PESTAÑAS----------------------------------------------

    # st.button("Log out", on_click=st.logout)


    # ---------------------------------------------------------------------------------------------------------------
    # Código de ventana modal con equipo
    # ---------------------------------------------------------------------------------------------------------------

    # 1. Definimos los datos de las 4 personas (puedes cambiar estos datos)
    personas = [
        {
            "nombre": "Isabel Rodriguez",
            "perfil": "Data Engineer",
            "foto": "https://media.licdn.com/dms/image/v2/D4D03AQEB9zs-8sm0kg/profile-displayphoto-shrink_800_800/B4DZWo_1IeG4Ac-/0/1742297061202?e=1781136000&v=beta&t=DlZB2b7KDVf7pbcSN6lfwn1ekKJEimhs93PVgWJV1WA",
            "linkedin": "https://www.linkedin.com/in/isabelrodriguezamor/"
        },
        {
            "nombre": "Iris Amorim",
            "perfil": "Data Scientist",
            "foto": "https://media.licdn.com/dms/image/v2/D4E03AQEJTs7nw_T3wA/profile-displayphoto-crop_800_800/B4EZ05mWVHIkAI-/0/1774787850946?e=1781136000&v=beta&t=hfzEhIwLN1YXl5NFHu2vRTmKDUeKyl4zTT1wKhEuoew",
            "linkedin": "https://www.linkedin.com/in/irisamorim/"
        },
        {
            "nombre": "Joaquin Lazaro",
            "perfil": "Data Scientist & Scrum Master",
            "foto": "https://media.licdn.com/dms/image/v2/D4E35AQFCyI20hWpOAA/profile-framedphoto-shrink_800_800/B4EZrj1PhzKMAg-/0/1764758976941?e=1780315200&v=beta&t=qlp2rAy0ITtc79TxXN5BqpMxHQA4iVvHNrL3na7rOpE",
            "linkedin": "https://www.linkedin.com/in/joaquin-lazarom/"
        },
        {
            "nombre": "Juan Manuel Iriondo",
            "perfil": "Data Analyst & Product Owner",
            "foto": "https://media.licdn.com/dms/image/v2/D4D35AQEXVlh9IvKGeg/profile-framedphoto-shrink_100_100/profile-framedphoto-shrink_100_100/0/1722427578700?e=1780311600&v=beta&t=7mi_4dkStCb5XVibTtIwN3bsdIKW76ybjBKwraKJv_A",
            "linkedin": "https://www.linkedin.com/in/juanmanueliriondoortega/"
        }
    ]

    # 2. Creamos la pantalla modal
    @st.dialog("Equipo de Trabajo", width="large")
    def mostrar_modal_tarjetas():
        st.write("Conoce a los miembros del equipo:")
        
        # Creamos 4 columnas dentro del modal para las 4 tarjetas
        columnas = st.columns(4)
        
        for i, persona in enumerate(personas):
            with columnas[i]:
                # Diseñamos la tarjeta usando HTML y CSS básico
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #e0e0e0; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center; 
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                        background-color: #f9f9f9;
                    ">
                        <img src="{persona['foto']}" alt="{persona['nombre']}" style="width: 80px; border-radius: 50%; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #333;">{persona['nombre']}</h4>
                        <h3 style="margin: 0; color: #333;">{persona['perfil']}</h3>
                        <br>
                        <a href="{persona['linkedin']}" target="_blank" style="
                            text-decoration: none; 
                            background-color: #0077B5; 
                            color: white; 
                            padding: 8px 15px; 
                            border-radius: 5px; 
                            font-weight: bold;
                            display: inline-block;
                        ">LinkedIn</a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )


    col21, col22 = st.columns([5, 1])
    with col21:
        st.button("Log out", on_click=st.logout)
    with col22:
        if st.button("👥 Equipo"):
            mostrar_modal_tarjetas()