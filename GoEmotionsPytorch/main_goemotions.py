import argparse
import json
import logging
import os
import glob
import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from GoEmotionsPytorch.model import BertForMultiLabelClassification
from GoEmotionsPytorch.utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from GoEmotionsPytorch.data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)


class EmotionDetectionClassification():
    def __init__(self):
        cli_parser = argparse.ArgumentParser()

        cli_parser.add_argument("--taxonomy", type=str, default='original', help="Taxonomy (original, ekman, group)")

        self.cli_args = cli_parser.parse_args()

        config_filename = "{}.json".format(self.cli_args.taxonomy)
        with open(os.path.join(os.path.dirname(__file__), "config", config_filename)) as f:
            args = AttrDict(json.load(f))
        logger.info("Training/evaluation parameters {}".format(args))

        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        init_logger()
        set_seed(args)

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)

    def train(self):
        # Read from config file and make args
        config_filename = "{}.json".format(self.cli_args.taxonomy)
        with open(os.path.join(os.path.dirname(__file__), "config", config_filename)) as f:
            args = AttrDict(json.load(f))
        logger.info("Training/evaluation parameters {}".format(args))

        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        init_logger()
        set_seed(args)

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)
        train_dataset, num_actual_predict_objects = load_and_cache_examples(args, tokenizer, mode="train", df=None) if args.train_file else None
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * args.warmup_proportion),
            num_training_steps=t_total
        )

        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total train batch size = %d", args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", args.logging_steps)
        logger.info("  Save steps = %d", args.save_steps)

        global_step = 0
        tr_loss = 0.0

        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                outputs = model(**inputs)

                loss = outputs[0]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        len(train_dataloader) <= args.gradient_accumulation_steps
                        and (step + 1) == len(train_dataloader)
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #     if args.evaluate_test_during_training:
                    #         evaluate(args, model, test_dataset, "test", global_step)
                    #     else:
                    #         evaluate(args, model, dev_dataset, "dev", global_step)

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to {}".format(output_dir))

                        if args.save_optimizer:
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

                if args.max_steps > 0 and global_step > args.max_steps:
                    break

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    def evaluate(self):
        mode = "test"
        config_filename = "{}.json".format(self.cli_args.taxonomy)
        with open(os.path.join(os.path.dirname(__file__), "config", config_filename)) as f:
            args = AttrDict(json.load(f))
        logger.info("Training/evaluation parameters {}".format(args))

        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        init_logger()
        set_seed(args)

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(os.path.join(os.path.dirname(__file__), args.output_dir) + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = BertForMultiLabelClassification.from_pretrained(checkpoint)
            model.to(args.device)
            test_dataset, num_actual_predict_objects = load_and_cache_examples(args, tokenizer, mode="test", df=None) if args.test_file else None
            eval_sampler = SequentialSampler(test_dataset)
            eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # Eval!
            if global_step != None:
                logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
            else:
                logger.info("***** Running evaluation on {} dataset *****".format(mode))
            logger.info("  Num examples = {}".format(len(test_dataset)))
            logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "labels": batch[3]
                    }
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if preds is None:
                    preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
                else:
                    preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            results = {
                "loss": eval_loss
            }
            preds[preds > args.threshold] = 1
            preds[preds <= args.threshold] = 0
            result = compute_metrics(out_label_ids, preds)
            results.update(result)

            output_dir = os.path.join(args.output_dir, mode)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
            with open(output_eval_file, "w") as f_w:
                logger.info("***** Eval results on {} dataset *****".format(mode))
                for key in sorted(results.keys()):
                    logger.info("  {} = {}".format(key, str(results[key])))
                    f_w.write("  {} = {}\n".format(key, str(results[key])))
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

    def evaluate_plot(self):
        mode = "test"
        config_filename = "{}.json".format(self.cli_args.taxonomy)
        with open(os.path.join(os.path.dirname(__file__), "config", config_filename)) as f:
            args = AttrDict(json.load(f))
        logger.info("Training/evaluation parameters {}".format(args))

        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        init_logger()
        set_seed(args)

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(os.path.join(os.path.dirname(__file__), args.output_dir) + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            result_dict = {
                'checkpoint': None,
                'loss': None,
                'accuracy': None,
            }
            for checkpoint in checkpoints:
                try:
                    csv_file = open(os.path.join(os.path.dirname(__file__), 'output', "eval_results.csv"), 'a')
                    writer = csv.DictWriter(csv_file, result_dict.keys())
                    global_step = checkpoint.split("-")[-1]
                    model = BertForMultiLabelClassification.from_pretrained(checkpoint)
                    model.to(args.device)
                    test_dataset, num_actual_predict_objects = load_and_cache_examples(args, tokenizer, mode="test", df=None) if args.test_file else None
                    eval_sampler = SequentialSampler(test_dataset)
                    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    # Eval!
                    if global_step != None:
                        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
                    else:
                        logger.info("***** Running evaluation on {} dataset *****".format(mode))
                    logger.info("  Num examples = {}".format(len(test_dataset)))
                    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
                    eval_loss = 0.0
                    nb_eval_steps = 0
                    preds = None
                    out_label_ids = None

                    for batch in tqdm(eval_dataloader, desc="Evaluating"):
                        model.eval()
                        batch = tuple(t.to(args.device) for t in batch)

                        with torch.no_grad():
                            inputs = {
                                "input_ids": batch[0],
                                "attention_mask": batch[1],
                                "token_type_ids": batch[2],
                                "labels": batch[3]
                            }
                            outputs = model(**inputs)
                            tmp_eval_loss, logits = outputs[:2]

                            eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_steps += 1
                        if preds is None:
                            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
                            out_label_ids = inputs["labels"].detach().cpu().numpy()
                        else:
                            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
                            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

                    eval_loss = eval_loss / nb_eval_steps
                    preds[preds > args.threshold] = 1
                    preds[preds <= args.threshold] = 0
                    result = compute_metrics(out_label_ids, preds)
                    result_dict = {
                        'checkpoint': os.path.basename(checkpoint).split('-')[1],
                        'loss': eval_loss,
                        'accuracy':  result["accuracy"],
                    }
                    writer.writerow(result_dict)
                    csv_file.close()
                except Exception as e:
                    pass
        # with open(os.path.join(os.path.dirname(__file__), 'output', "eval_results.csv"), 'a') as f:
        #     result_dict = {
        #         'checkpoint': None,
        #         'loss': None,
        #         'accuracy': None,
        #     }
        #     writer = csv.DictWriter(f, result_dict.keys())
        #     for index in range(10):
        #         result_dict = {
        #             'checkpoint': 'test_1',
        #             'loss': 'test_2',
        #             'accuracy': 'test_3',
        #         }
        #         writer.writerow(result_dict)

    def predict(self, df):
        mode = "pred"
        config_filename = "{}.json".format(self.cli_args.taxonomy)
        with open(os.path.join(os.path.dirname(__file__), "config", config_filename)) as f:
            args = AttrDict(json.load(f))
        logger.info("Training/evaluation parameters {}".format(args))

        args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

        init_logger()
        set_seed(args)

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )

        # GPU or CPU
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        model.to(args.device)
        pred_dataset, num_actual_predict_objects = load_and_cache_examples(args, tokenizer, mode="pred", df=df) if args.test_file else None
        emotion_file_path = os.path.join(os.path.dirname(__file__), 'data', 'emotions.txt')
        with open(emotion_file_path, "r") as f:
            all_emotions = f.read().splitlines()
            idx2emotion = {i: e for i, e in enumerate(all_emotions)}
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(os.path.join(os.path.dirname(__file__), args.output_dir) + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        checkpoint = checkpoints[-1]
        global_step = checkpoint.split("-")[-1]
        model = BertForMultiLabelClassification.from_pretrained(checkpoint)
        model.to(args.device)
        results = {}
        eval_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        logger.info("  Num examples = {}".format(len(pred_dataset)))
        logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(pred_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        emotions_columns = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval",
                            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
                            "relief", "remorse", "sadness", "surprise", "neutral", 'top_1_emotion', 'top_1_prob', 'top_2_emotion', 'top_2_prob',
                            'top_3_emotion', 'top_3_prob']
        df = self.updateDFWithPredictions(df, preds, num_actual_predict_objects, idx2emotion, emotions_columns)
        prediction_results_file_path = os.path.join(os.path.dirname(__file__), 'output', 'prediction_results.csv')
        df.to_csv(prediction_results_file_path, index=False)


    def updateDFWithPredictions(self, df, result, num_actual_predict_objects, idx2emotion, emotions_columns):
        df_pred = pd.DataFrame()
        for (i, probabilities) in enumerate(result):
            sorted_idx = np.argsort(-probabilities)
            probabilities_df = pd.DataFrame(probabilities).T
            probabilities_df['top_1_emotion'] = idx2emotion[sorted_idx[0]]
            probabilities_df['top_1_prob'] = probabilities[sorted_idx[0]]
            probabilities_df['top_2_emotion'] = idx2emotion[sorted_idx[1]]
            probabilities_df['top_2_prob'] = probabilities[sorted_idx[1]]
            probabilities_df['top_3_emotion'] = idx2emotion[sorted_idx[2]]
            probabilities_df['top_3_prob'] = probabilities[sorted_idx[2]]
            df_pred = df_pred.append(probabilities_df)
            if i >= num_actual_predict_objects:
              break
        df_pred.columns = emotions_columns
        df_pred = df_pred.reset_index(drop=True)
        df = pd.concat([df, df_pred], axis=1)
        return df


if __name__ == '__main__':
    emotionDetectionClassification = EmotionDetectionClassification()
    emotionDetectionClassification.evaluate_plot()

