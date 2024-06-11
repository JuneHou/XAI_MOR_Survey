from sentence_transformers import SentenceTransformer, util
from IPython.display import display, HTML
import pandas as pd
import torch
import spacy
import scispacy
import medspacy
from medspacy.ner import TargetRule
import re

class SBERTQuery:
    def __init__(self, train_admission, test_admission, embedder):
        self.embedder = embedder
        self.train_admission = train_admission
        self.corpus = self.train_admission.text
        self.corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)
        self.test_admission = test_admission

    def balanced_queries(self):
        positive = self.test_admission[self.test_admission['hospital_expire_flag'] == 1].sample(n=5)
        negative = self.test_admission[self.test_admission['hospital_expire_flag'] == 0].sample(n=5)
        combined_instances = pd.concat([positive, negative]).reset_index(drop=True)
        queries = combined_instances.text
        return queries

    def top_k_queries(self, queries, top_k=3):
        # Find the closest sentences of the corpus for each query sentence based on cosine similarity
        for query in queries:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 3 most similar sentences in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                idx_value = idx.item()
                print("\nRetrieved label: ", self.train_admission.hospital_expire_flag[idx_value])
                print(self.train_admission.text[idx_value], "\n(Score: {:.4f})".format(score.item()))

    def select_instance(self, k=10):
        queries = self.test_admission.text
        select_list = []
        select_label = []

        for i in range(len(queries)):
            label = self.test_admission.hospital_expire_flag[i]
            query = queries[i]
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)

            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k)

            idx_list = top_results[1]
            flag = True
            for idx in idx_list:
                if self.train_admission.hospital_expire_flag[idx.item()] != label:
                    flag = False
                    break

            if flag == True:
                select_list.append(i)
                select_label.append(label)

        return select_list, select_label

    def med7_ner(self, text):
        med7 = spacy.load("en_core_med7_trf")
        doc = med7(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def scispacy_ner(self, text):
        nlp = spacy.load("en_ner_bc5cdr_md")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def medspacy_ner(self, text):
        nlp = medspacy.load("en_core_med7_trf")
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def query_ner(self, queries_idx, top_k, ner_model='scispacy'):
        if ner_model == 'scispacy':
            ner_model = self.scispacy_ner
        elif ner_model == 'medspacy':
            ner_model = self.medspacy_ner
        else:
            ner_model = self.med7_ner
        
        results_dict = {query: {"retrieved_texts": [], "retrieved_labels": [], "retrieved_scores": [], "matched_ents":[], "unmatched_ents":[]} for query in queries_idx}

        for query_idx in queries_idx:
            query = self.test_admission.text[query_idx]
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            for score, idx in zip(top_results[0], top_results[1]):
                idx_value = idx.item()
                results_dict[query_idx]["retrieved_labels"].append(self.train_admission.hospital_expire_flag[idx_value])
                results_dict[query_idx]["retrieved_texts"].append(self.train_admission.text[idx_value])
                results_dict[query_idx]["retrieved_scores"].append(score.item())

            # Apply NER for both query and retrieved texts
            query_entities = ner_model(query)
            q_entity_values_lower = [(ent[0][0].lower() + ent[0][1:]) for ent in query_entities]  # Lowercase only the first character
            retrieved_entities = [ner_model(text) for text in results_dict[query_idx]["retrieved_texts"]]

            # Find the matched NER words or subwords
            for r_entities in retrieved_entities:
                r_entity_values_lower = [(ent[0][0].lower() + ent[0][1:]) for ent in r_entities]  # Same here

                matched = [ent for ent in r_entity_values_lower if ent in q_entity_values_lower]
                unmatched = [ent for ent in r_entity_values_lower if ent not in matched]
                results_dict[query_idx]["matched_ents"].append(matched)
                results_dict[query_idx]["unmatched_ents"].append(unmatched)
            
            # Query matched and unmatched
            all_matched = [item for sublist in results_dict[query_idx]['matched_ents'] for item in sublist]
            all_unmatched = [ent for ent in q_entity_values_lower if ent not in all_matched]

            results_dict[query_idx]["matched_ents"].insert(0, all_matched)
            results_dict[query_idx]["unmatched_ents"].insert(0, all_unmatched)

        return results_dict

    def visualize(self, queries_idx, result_dict):
        data = result_dict[queries_idx]

        # Add the query text to the retrieved_texts list
        all_texts = [self.test_admission.text[queries_idx]] + data['retrieved_texts']

        highlighted_texts = []

        # Function to apply highlighting
        def apply_highlight(text, entity, color):
            if entity:  # Check if entity is not empty
                first_char = re.escape(entity[0])
                rest = re.escape(entity[1:])
                pattern = rf'\b{first_char}(?i){rest}\b'

                def replacer(match):
                    original = match.group(0)  # Get the matched string with original case
                    return f'<span style="background-color: {color}">{original}</span>'

                text = re.sub(pattern, replacer, text)
            return text

        # Iterate through all texts (retrieved + query) and their corresponding matched and unmatched entities
        for i, (text, matched_ents, unmatched_ents) in enumerate(zip(all_texts, data["matched_ents"], data["unmatched_ents"])):
            # Highlight matched entities
            for match in set(matched_ents):  # Using set to avoid duplicates
                text = apply_highlight(text, match, 'LightSalmon')

            # Highlight unmatched entities
            for unmatch in set(unmatched_ents):
                text = apply_highlight(text, unmatch, 'PowderBlue')

            highlighted_texts.append(text.replace("\n", "<br>"))

        i=1
        # Output the highlighted texts
        for text in highlighted_texts:
            display(HTML(text))
            if i <=3 :
                print('-'*18, 'Retrieved Patient: ', i, '-'*18)
                i+=1
