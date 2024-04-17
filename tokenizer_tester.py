import pandas as pd
from transformers import AutoTokenizer

def load_and_tokenize_data(csv_file, tokenizer_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    df = pd.read_csv(csv_file)
    if "content" not in df.columns:
        raise ValueError("La columna 'content' no se encontró en el CSV.")

    tokenized_data = tokenizer(df["content"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Retornar tanto los datos tokenizados como el tokenizer
    return tokenized_data, tokenizer

def main():
    csv_file = "C:\\Users\\lucas\\Documents\\virtual Dev\\starcoder2 test\\formatted_code_dataset.csv"
    tokenizer_checkpoint = "bigcode/starcoder2-3b"

    # Asegúrate de desempacar tanto tokenized_data como tokenizer aquí
    tokenized_data, tokenizer = load_and_tokenize_data(csv_file, tokenizer_checkpoint)

    N = 5
    for i in range(N):
        print(f"Ejemplo {i+1}:")
        print(tokenized_data["input_ids"][i])
        print(tokenized_data["attention_mask"][i])
        # Ahora puedes usar tokenizer aquí sin problemas
        print("Tokens:", tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][i]))
        print("-" * 50)

if __name__ == "__main__":
    main()
