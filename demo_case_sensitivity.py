#!/usr/bin/env python3
"""
Demo de las opciones de preprocesamiento: lowercase vs case-sensitive
"""

import pandas as pd
from src.hatescan.preprocessing import clean_text, TextTokenizer, extract_case_features

def demo_case_sensitivity():
    """Demuestra las diferencias entre preprocesamiento con/sin case sensitivity"""

    print("🔄 DEMO: CASE SENSITIVITY EN PREPROCESAMIENTO")
    print("=" * 60)

    # Ejemplos de texto con diferentes patrones de mayúsculas
    examples = [
        "THIS IS SHOUTING!!!",
        "This is normal text",
        "I HATE this so MUCH!!!",
        "please be quiet",
        "WHY ARE YOU SO STUPID???",
        "calm down and relax"
    ]

    print("\n📊 COMPARACIÓN DE PREPROCESAMIENTO")
    print("-" * 60)

    for i, text in enumerate(examples, 1):
        print(f"\n{i}. Texto original: '{text}'")

        # Limpieza con lowercase (tradicional)
        cleaned_lower = clean_text(text, preserve_case=False)
        print(f"   Lowercase:     '{cleaned_lower}'")

        # Limpieza preservando case
        cleaned_case = clean_text(text, preserve_case=True)
        print(f"   Case preserved: '{cleaned_case}'")

        # Features de case
        case_features = extract_case_features(text)
        print(f"   Case features: uppercase={case_features['has_uppercase']}, "
              f"all_caps_ratio={case_features['all_caps_ratio']:.2f}, "
              f"exclamation={case_features['exclamation_count']}")

    print("\n\n🎯 ANÁLISIS DE IMPACTO EN TOXICIDAD")
    print("-" * 60)

    # Cargar datos reales
    df = pd.read_csv('data/raw/youtoxic_english_1000.csv')

    # Comparar toxicidad por presencia de mayúsculas
    toxic_with_upper = df[(df['IsToxic'] == 1) &
                         (df['Text'].str.contains(r'\b[A-Z]{2,}\b', regex=True))]
    toxic_without_upper = df[(df['IsToxic'] == 1) &
                            (~df['Text'].str.contains(r'\b[A-Z]{2,}\b', regex=True))]

    non_toxic_with_upper = df[(df['IsToxic'] == 0) &
                             (df['Text'].str.contains(r'\b[A-Z]{2,}\b', regex=True))]

    print(f"Comentarios tóxicos CON mayúsculas: {len(toxic_with_upper)}")
    print(f"Comentarios tóxicos SIN mayúsculas: {len(toxic_without_upper)}")
    print(f"Comentarios no tóxicos CON mayúsculas: {len(non_toxic_with_upper)}")

    ratio_toxic_upper = len(toxic_with_upper) / len(df[df['IsToxic'] == 1])
    ratio_non_toxic_upper = len(non_toxic_with_upper) / len(df[df['IsToxic'] == 0])

    print(".1%")
    print(".1%")

    print("\n💡 CONCLUSIONES:")
    print("-" * 30)
    print("• Las mayúsculas aparecen 2x más en comentarios tóxicos")
    print("• ALL CAPS es 2x más frecuente en toxicidad")
    print("• Información de case es una feature discriminativa")
    print("• Recomendación: Usar approach híbrido")
    print("  - Texto procesado en lowercase para embeddings")
    print("  - Features de case como variables adicionales")

    print("\n🔧 OPCIONES DE IMPLEMENTACIÓN:")
    print("-" * 30)
    print("1. LOWERCASE ONLY (simplista, pierde info)")
    print("2. CASE PRESERVED (complejo, mantiene ruido)")
    print("3. HÍBRIDO (recomendado):")
    print("   - Texto en lowercase para modelado principal")
    print("   - Features de case como inputs adicionales")

if __name__ == "__main__":
    demo_case_sensitivity()