#!/usr/bin/env python3
"""
Demo del flujo completo de HateScan: Scraper → Cleaner → Tokenizer → Modelo
"""

import pandas as pd
from src.hatescan.preprocessing import clean_text, TextTokenizer
from src.hatescan.scraping.youtube_scraper import fetch_comments, save_to_csv
import os

def demo_flujo_completo():
    """Demuestra el flujo completo desde scraping hasta preprocesamiento"""

    print("🔄 DEMO DEL FLUJO COMPLETO DE HATESCAN")
    print("=" * 50)

    # 1. SCRAPING: Obtener comentarios de YouTube
    print("\n📺 PASO 1: SCRAPING DE YOUTUBE")
    print("-" * 30)

    # Usar el mismo video que ya tenemos datos
    video_id = "dQw4w9WgXcQ"

    # Verificar si ya tenemos datos scrapeados
    scraped_file = f"data/raw/comments_{video_id}_20260511.csv"
    if os.path.exists(scraped_file):
        print(f"✅ Usando datos ya scrapeados: {scraped_file}")
        df_scraped = pd.read_csv(scraped_file)
    else:
        print(f"🔄 Scrapeando video {video_id}...")
        raw_comments = fetch_comments(video_id, max_results=5)
        if raw_comments:
            save_to_csv(raw_comments, f"comments_{video_id}_20260511.csv")
            df_scraped = pd.read_csv(scraped_file)
        else:
            print("❌ Error en scraping")
            return

    print(f"📊 Comentarios scrapeados: {len(df_scraped)}")
    print("Comentarios de ejemplo:")
    for i, row in df_scraped.head(3).iterrows():
        print(f"  {i+1}. {row['comment'][:100]}...")

    # 2. CLEANING: Limpiar texto
    print("\n🧹 PASO 2: LIMPIEZA DE TEXTO")
    print("-" * 30)

    df_scraped['text_clean'] = df_scraped['comment'].apply(clean_text)

    print("Ejemplos de limpieza:")
    for i, row in df_scraped.head(3).iterrows():
        print(f"  Original: {row['comment'][:80]}...")
        print(f"  Limpio:   {row['text_clean'][:80]}...")
        print()

    # 3. TOKENIZATION: Tokenizar y lematizar
    print("\n🔤 PASO 3: TOKENIZACIÓN Y LEMATIZACIÓN")
    print("-" * 30)

    tokenizer = TextTokenizer()
    df_scraped['tokens'] = df_scraped['text_clean'].apply(tokenizer.tokenize)
    df_scraped['text_processed'] = df_scraped['text_clean'].apply(tokenizer.process)

    print("Ejemplos de tokenización:")
    for i, row in df_scraped.head(3).iterrows():
        print(f"  Texto limpio: {row['text_clean'][:60]}...")
        print(f"  Tokens: {row['tokens'][:10]}...")  # Solo primeros 10 tokens
        print(f"  Procesado: {row['text_processed'][:60]}...")
        print()

    # 4. COMPARACIÓN CON DATASET DE ENTRENAMIENTO
    print("\n📚 PASO 4: COMPARACIÓN CON DATASET DE ENTRENAMIENTO")
    print("-" * 50)

    # Cargar dataset de entrenamiento
    df_train = pd.read_csv('data/raw/youtoxic_english_1000.csv')

    print(f"📊 Dataset entrenamiento: {len(df_train)} comentarios")
    print(f"📊 Comentarios scrapeados: {len(df_scraped)} comentarios")

    # Aplicar mismo preprocesamiento al dataset de entrenamiento
    df_train['text_clean'] = df_train['Text'].apply(clean_text)
    df_train['text_processed'] = df_train['text_clean'].apply(tokenizer.process)

    print("\nEstadísticas del dataset de entrenamiento:")
    print(f"  - Comentarios tóxicos: {df_train['IsToxic'].sum()}")
    print(f"  - Comentarios no tóxicos: {len(df_train) - df_train['IsToxic'].sum()}")
    print(".1f")

    # Mostrar ejemplos del dataset de entrenamiento
    print("\nEjemplos del dataset de entrenamiento:")
    toxic_examples = df_train[df_train['IsToxic'] == 1].head(2)
    for i, row in toxic_examples.iterrows():
        print(f"  Tóxico: {row['text_processed'][:80]}...")

    non_toxic_examples = df_train[df_train['IsToxic'] == 0].head(2)
    for i, row in non_toxic_examples.iterrows():
        print(f"  No tóxico: {row['text_processed'][:80]}...")

    # 5. ANÁLISIS DEL IMPACTO DEL PREPROCESAMIENTO
    print("\n📈 PASO 5: ANÁLISIS DEL IMPACTO DEL PREPROCESAMIENTO")
    print("-" * 50)

    # Calcular reducción de longitud
    df_scraped['len_original'] = df_scraped['comment'].str.len()
    df_scraped['len_clean'] = df_scraped['text_clean'].str.len()
    df_scraped['len_processed'] = df_scraped['text_processed'].str.len()
    df_scraped['num_tokens'] = df_scraped['tokens'].str.len()

    print("Estadísticas de reducción:")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")

    print("\n✅ FLUJO COMPLETADO EXITOSAMENTE")
    print("El pipeline está listo para:")
    print("  1. Scraper → Limpieza → Tokenización")
    print("  2. Entrenamiento de modelos con datos procesados")
    print("  3. Inferencia en tiempo real con mismo preprocesamiento")

if __name__ == "__main__":
    demo_flujo_completo()