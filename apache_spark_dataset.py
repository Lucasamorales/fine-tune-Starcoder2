from preprocess import format_dataframe

import requests
import os
import glob
import pandas as pd
from datasets import load_dataset


def download_apache_spark_source_code():
    """
        Descarga el código fuente de Apache Spark desde el repositorio oficial en GitHub.
        Almacena el archivo zip localmente y luego lo extrae.
    """

    # URL del repositorio de GitHub de Apache Spark (zip del código fuente)
    url = "https://github.com/apache/spark/archive/refs/heads/master.zip"
    local_zip_path = "spark-master.zip"
    # Descarga del archivo zip
    try:
        response = requests.get(url)
        with open(local_zip_path, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        print(f"Error downloading or writing Apache Spark source code: {e}")
        return
    # Descompresión del código fuente descargado
    import zipfile
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall("spark-source")
        print("Código fuente de Apache Spark descargado y extraído.")
    except zipfile.BadZipFile:
        print("Fallo al descomprimir el código fuente de Apache Spark. El archivo descargado puede estar corrupto.")


def prepare_dataset_for_training(directory_path):
    """
       Prepara el dataset para entrenamiento buscando archivos de código fuente con extensiones específicas,
       leyendo su contenido, y aplicando un procesamiento básico.
       Argumentos:
           directory_path (str): Ruta al directorio donde se encuentra el código fuente extraído.
       Salida:
           None, pero genera un archivo CSV con el código fuente formateado.
    """
    # Dictionary to map file extensions to programming languages
    extension_to_language = {
        '.scala': 'SCALA',
        '.java': 'JAVA',
        '.py': 'PYTHON',
        '.sql': 'SQL'
    }

    all_data = []

    # Encuentra archivos con extensiones específicas
    for extension, language in extension_to_language.items():
        file_paths = glob.glob(f"{directory_path}/**/*{extension}", recursive=True)
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                file_size = os.path.getsize(file_path)
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                    file_size = os.path.getsize(file_path)
                except Exception as e:
                    print(f"Error al procesar {file_path} con codificación latin-1: {e}")
                    continue
            except Exception as e:
                print(f"Error al procesar {file_path} con codificación utf-8: {e}")
                continue

            processed_content = format_dataframe(content)
            all_data.append({
                'content': processed_content['content'],
                'encoding': processed_content['encoding'],
                'avg_line_length': processed_content['avg_line_length'],
                'max_line_length': processed_content['max_line_length'],
                'alphanum_fraction': processed_content['alphanum_fraction'],
                'path': file_path,
                'size': file_size,
                'language': language
            })

    df = pd.DataFrame(all_data)
    df = df.sort_values(by='language')  # Sort the DataFrame by language
    df.to_csv("formatted_code_dataset.csv", index=False, escapechar="\\")
    print("Code data formatted and saved as formatted_code_dataset.csv")

directory_path = "C:\\Users\\lucas\\Documents\\virtual Dev\\starcoder2 test\\spark-source"
prepare_dataset_for_training(directory_path)

