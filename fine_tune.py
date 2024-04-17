import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
       Función principal para configurar y entrenar el modelo StarCoder2 utilizando un dataset específico.
       Realiza el finetuning del modelo con control de errores y registro de métricas.
    """
    # Asegura que las credenciales de wandb y HuggingFace Hub estén configuradas
    if "WANDB_API_KEY" not in os.environ or "HF_TOKEN" not in os.environ:
        logger.error("Credenciales de Wandb o HuggingFace no encontradas. Establezca las variables de entorno correspondientes.")
        return

    # Configuración del modelo y tokenizer
    checkpoint = "bigcode/starcoder2-3b"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, pad_token='<PAD>')
    # Si el tokenizer no tiene un token de pad, se establece al token EOS
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # Carga y preparación del dataset
    full_dataset = load_dataset("bigcode/the-stack-smol", split="train")
    dataset_size = len(full_dataset)
    train_size = int(0.15 * dataset_size)
    dataset = full_dataset.shuffle(seed=42).select(range(train_size))

    # Función para tokenizar los ejemplos del dataset
    def tokenize_function(examples):
        """
               Tokeniza el contenido del código, ajustando la tokenización para manejar errores de CUDA
               y reduciendo la longitud máxima de tokenización para evitar errores.
        """

        try:
            return tokenizer(examples["content"], padding="max_length", truncation=True,
                             max_length=32)  #
        except Exception as e:
            logger.error(f"Error durante la tokenización: {str(e)}"")
            raise e

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Configuración de los argumentos para el entrenamiento del modelo
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=f"{checkpoint}",
        no_cuda=False,
    )

    # Inicialización del objeto Trainer con los argumentos y el dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
    )

    # Bloqueo de lanzamiento de CUDA para detallar errores
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Comienzo del entrenamiento con manejo de errores
    try:
        for epoch in range(int(training_args.num_train_epochs)):
            model.train()
            total_loss = 0
            for batch in trainer.get_train_dataloader():
                batch = {k: v.to(model.device) for k, v in batch.items()}   # Asegura que el lote esté en el dispositivo correcto
                outputs = model(**batch)
                loss = outputs.loss
                if loss is not None:
                    loss.backward()
                    trainer.optimizer.step()
                    trainer.lr_scheduler.step()
                    trainer.optimizer.zero_grad()
                    total_loss += loss.item()
            avg_loss = total_loss / len(trainer.get_train_dataloader())
            logger.info(f"Epoch {epoch + 1}:  Pérdida promedio:: {avg_loss}")
            # Evaluación del modelo después de cada época
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in trainer.get_eval_dataloader():
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
            avg_eval_loss = eval_loss / len(trainer.get_eval_dataloader())
            logger.info(f"Epoch {epoch + 1}: Pérdida de evaluación: {avg_eval_loss}")

        logger.info("Ajuste la configuración del modelo o de entrenamiento según sea necesario."")
    except RuntimeError as e:
        logger.error(f"Runtime error during training: {str(e)}")
        if "CUDA error: device-side assert triggered" in str(e):
            logger.error(
                "This CUDA error might be due to incorrect dataset formatting or an issue with tokenization. Enabling CUDA_LAUNCH_BLOCKING=1 for detailed stack trace.")
        elif "CUDA out of memory" in str(e):
            logger.error("CUDA out of memory error: Try reducing the batch size or sequence length.")
        os.environ[
            "CUDA_LAUNCH_BLOCKING"] = "0"   # Reinicio de CUDA_LAUNCH_BLOCKING tras el entrenamiento para evitar impactos en el rendimiento
        return
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        return
    logger.info("Revise la configuración del entrenamiento y del modelo.")


# trainer.push_to_hub()

if __name__ == "__main__":
    main()
