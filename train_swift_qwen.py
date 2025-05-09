import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse

logger = get_logger()
seed_everything(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Model name, e.g. qwen3-32b')

    args = parser.parse_args()

    model_id_or_path = args.model_name # model_id or model_path
    sys_prompt = open("prompt_bank/pure_text_infer.txt", "r", encoding="utf-8").read()
    output_dir = f'{model_id_or_path}_lora_test'

    # dataset
    dataset = ["dataset/train.jsonl"]  # dataset_id or dataset_path
    data_seed = 42
    max_length = 8192
    split_dataset_ratio = 0.001  # Split validation set
    num_proc = 4  # The number of processes for data loading.

    # lora
    lora_rank, lora_alpha = 16, 32

    # training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=100,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f'output_dir: {output_dir}')


    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_tokenizer(model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer, default_system=sys_prompt, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(task_type='CAUSAL_LM',
                             r=lora_rank,
                             lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Load the dataset, split it into a training set and a validation set,
    train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc, seed=data_seed)
    logger.info(f'train_dataset[0]: {train_dataset[0]}')

    # Encode the text data into tokens.
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')