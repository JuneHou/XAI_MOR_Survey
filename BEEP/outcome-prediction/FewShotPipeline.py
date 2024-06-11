import guidance
import openai
import pandas as pd
import numpy as np
import torch
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from XAI_Utils import MOR_Predictor
from time import sleep

class FewShotPipeline:

    def __init__(self, Predictor, train_admission, test_admission, program, num_samples):
        self.Predictor = Predictor
        self.train_admission = train_admission
        self.test_admission = test_admission
        self.train_embeddings = None
        self.selected_samples = None
        self.program = program
        self.num_samples = num_samples

    def clean_data(self, data):

         # Remove Handlebar syntax
        data['text'] = data['text'].str.replace("{{", "").str.replace("}}", "")
        return data

    def create_embeddings(self):
        self.train_embeddings = [self.Predictor.cls_embed(text) for text in self.train_admission.text]
        self.train_embeddings = np.array(self.train_embeddings)

    def select_samples(self, random_state=24):
        positive = self.test_admission[self.test_admission['hospital_expire_flag'] == 1].sample(n=self.num_samples, random_state=random_state)
        negative = self.test_admission[self.test_admission['hospital_expire_flag'] == 0].sample(n=self.num_samples, random_state=random_state)
        selected = pd.concat([positive, negative])
        # Clean the selected samples
        self.selected_samples = self.clean_data(selected)

    def setup_model(self):
        guidance.llm = guidance.llms.OpenAI("gpt-4")
        guidance.llms.Transformers.cache.clear()

    def compute_metrics(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        return accuracy, precision, recall

    def predict(self):
        id_list = self.selected_samples.index
        results = {
            "idx_list": [],
            "merged_idx_list": [],
            "predicted_list": [],
            "true_labels": [],
            "errors": []  # This will store dictionaries of error information
        }

        for idx in id_list:
            instance = self.test_admission.iloc[idx, 1]
            target_label = self.test_admission.iloc[idx, -1]
            target_label_desc = self.Predictor.convert_label(target_label)  # Convert to descriptive label
            merged_indexes, prompt_list = self.Predictor.knn_prompt(self.train_embeddings, self.train_admission, instance, target_label, 'cosine')
            prediction = self.program(prompt_list=prompt_list, test_prompt=instance, target_label=target_label_desc)  # Use descriptive label
            sleep(60)

            if 'answer' in prediction:
                results["idx_list"].append(idx)
                predicted_label_desc = prediction['answer'].strip().lower()  # Ensure the label is in the correct format for comparison
                results["predicted_list"].append(predicted_label_desc)
                results["true_labels"].append(target_label_desc)
                results["merged_idx_list"].append(merged_indexes)
            else:
                error_info = {
                    "index": idx,
                    "label": target_label_desc,  # Use descriptive label
                    "neighbors": merged_indexes
                }
                results["errors"].append(error_info)

        return results
    

    def predict_manual(self, id_list):
        results = {
            "idx_list": [],
            "merged_idx_list": [],
            "predicted_list": [],
            "true_labels": [],
            "errors": []  # This will store dictionaries of error information
        }

        for idx in id_list:
            instance = self.test_admission.iloc[idx, 1]
            target_label = self.test_admission.iloc[idx, -1]
            merged_indexes, prompt_list = self.Predictor.knn_prompt(self.train_embeddings, self.train_admission, instance, target_label, 'cosine')
            prediction = self.program(prompt_list=prompt_list, test_prompt=instance, target_label=target_label)
            sleep(10)

            if 'answer' in prediction:
                results["idx_list"].append(idx)
                results["predicted_list"].append(int(prediction['answer']))
                results["true_labels"].append(target_label)
                results["merged_idx_list"].append(merged_indexes)
            else:
                error_info = {
                    "index": idx,
                    "label": target_label,
                    "neighbors": merged_indexes
                }
                results["errors"].append(error_info)

        return results

