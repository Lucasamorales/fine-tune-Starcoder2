import pandas as pd
import re


def format_dataframe(content):
    """
         formatea el contenido del archivo fuente eliminando comentarios y espacios adicionales.
        Además, calcula la longitud media y máxima de las líneas y la fracción alfanumérica del contenido.
        Argumentos:
            content (str): Contenido del archivo fuente como una sola cadena de texto.
        Salida:
            dict: Un diccionario con el contenido limpio y las métricas calculadas.
    """
    # Elimina comentarios de una línea
    content = re.sub(r"//.*", "", content)
    # Elimina comentarios multilínea
    content = re.sub(r"/\*[\s\S]*?\*/", "", content)

    # Elimina espacios en blanco al inicio y final de cada línea
    lines = content.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines and strip spaces
    cleaned_content = "\n".join(cleaned_lines)

    # Calcula la longitud promedio y máxima de las líneas
    line_lengths = [len(line) for line in cleaned_lines]
    avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    max_line_length = max(line_lengths) if line_lengths else 0

    # Calcula la fracción alfanumérica
    alphanum_chars = sum(c.isalnum() for c in cleaned_content)
    total_chars = len(cleaned_content)
    alphanum_fraction = alphanum_chars / total_chars if total_chars else 0

    return {
        'content': cleaned_content,
        'encoding': 'utf-8',
        'avg_line_length': avg_line_length,
        'max_line_length': max_line_length,
        'alphanum_fraction': alphanum_fraction
    }