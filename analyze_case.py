import pandas as pd

df = pd.read_csv('data/raw/youtoxic_english_1000.csv')

toxic_comments = df[df['IsToxic'] == 1]['Text']
non_toxic_comments = df[df['IsToxic'] == 0]['Text']

def analyze_case_usage(texts):
    total_words = 0
    uppercase_words = 0
    all_caps_words = 0

    for text in texts:
        words = str(text).split()
        total_words += len(words)

        for word in words:
            if word.isupper() and len(word) > 1:
                uppercase_words += 1
                if len(word) > 2:
                    all_caps_words += 1

    return {
        'total_words': total_words,
        'uppercase_words': uppercase_words,
        'all_caps_words': all_caps_words,
        'uppercase_ratio': uppercase_words / total_words if total_words > 0 else 0,
        'all_caps_ratio': all_caps_words / total_words if total_words > 0 else 0
    }

toxic_stats = analyze_case_usage(toxic_comments)
non_toxic_stats = analyze_case_usage(non_toxic_comments)

print('ANALISIS DE USO DE MAYUSCULAS EN EL DATASET')
print('=' * 50)
print(f'TOXICOS ({len(toxic_comments)} comentarios):')
print(f'  - Palabras en mayusculas: {toxic_stats["uppercase_words"]}')
print(f'  - Ratio mayusculas: {toxic_stats["uppercase_ratio"]:.3f}')
print(f'  - Palabras ALL CAPS: {toxic_stats["all_caps_words"]}')
print(f'  - Ratio ALL CAPS: {toxic_stats["all_caps_ratio"]:.3f}')
print()
print(f'NO TOXICOS ({len(non_toxic_comments)} comentarios):')
print(f'  - Palabras en mayusculas: {non_toxic_stats["uppercase_words"]}')
print(f'  - Ratio mayusculas: {non_toxic_stats["uppercase_ratio"]:.3f}')
print(f'  - Palabras ALL CAPS: {non_toxic_stats["all_caps_words"]}')
print(f'  - Ratio ALL CAPS: {non_toxic_stats["all_caps_ratio"]:.3f}')
print()
print('DIFERENCIAS:')
print(f'  - Ratio mayusculas (toxico - no toxico): {toxic_stats["uppercase_ratio"] - non_toxic_stats["uppercase_ratio"]:.3f}')
print(f'  - Ratio ALL CAPS (toxico - no toxico): {toxic_stats["all_caps_ratio"] - non_toxic_stats["all_caps_ratio"]:.3f}')

# Mostrar ejemplos concretos
print()
print('EJEMPLOS CONCRETOS:')
print('=' * 30)

print('Comentarios TOXICOS con mayusculas:')
toxic_upper = []
for text in toxic_comments.head(10):
    words = str(text).split()
    if any(word.isupper() and len(word) > 1 for word in words):
        toxic_upper.append(str(text)[:100])

for i, example in enumerate(toxic_upper[:3]):
    print(f'{i+1}. {example}...')

print()
print('Comentarios NO TOXICOS con mayusculas:')
non_toxic_upper = []
for text in non_toxic_comments.head(10):
    words = str(text).split()
    if any(word.isupper() and len(word) > 1 for word in words):
        non_toxic_upper.append(str(text)[:100])

for i, example in enumerate(non_toxic_upper[:3]):
    print(f'{i+1}. {example}...')