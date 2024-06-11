import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import openai
import guidance

class MOR_Predictor():

    def __init__(self, model, tokenizer, MAX_LEN, device):
        self.mp_model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def tokens_to_features(self, tokens):
        # Convert tokens to features
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding_length = self.MAX_LEN - len(input_ids)

        # Pad the input_ids and attention_mask
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor([input_ids]).to(self.device),
            'attention_mask': torch.tensor([attention_mask]).to(self.device)
        }

    def preprocess_function(self, instance):
        # Tokenize the instance
        tokens = self.tokenizer.tokenize(instance)
        # Truncate or pad the tokens if they exceed the model's maximum sequence length
        tokens = tokens[:self.MAX_LEN]
        # Convert tokens to features
        features = self.tokens_to_features(tokens)
        return features

    def predict_proba(self, texts):
        all_predictions = []
        for text in texts:
            # Preprocess the instance
            features = self.preprocess_function(text)

            # Forward pass through the model
            with torch.no_grad():
                outputs = self.mp_model(**features)
                logits = outputs.logits
                proba = F.softmax(logits, dim=-1).squeeze().cpu().detach().numpy()

            all_predictions.append(proba)

        return np.array(all_predictions)
    
    def attention(self, instance):
        # Preprocess the instance
        features = self.preprocess_function(instance)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.mp_model(**features)
            attentions = outputs.attentions

        # Convert attentions to numpy
        inputs = self.tokenizer(
                                instance, 
                                add_special_tokens=True,
                                max_length=self.MAX_LEN,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt'
                            )
        input_ids = inputs.input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        return attentions, tokens

    def agg_subwords(self, tokens):

        # In the list of tokens, if the token has "##" pattern, then the previous word of
        # this token is the start of the mask, which "i" should be added into the list of 
        # "agg_tokens". However, the idx of token has masked pattern is skipped.
        # In the attention, for the [CLS] in first setence,
        # only take the its attention weight which idx is in the "agg_tokens" list
        # word_start is used to store the token, in case it is the start of the masked word
        # flag is the indicator for whether current token is a subword, and will help to aggregate
        # subwords after the '##+' token to normal word
        # If the token is not '##+' aggregate word and append to list, flag = False

        agg_tokens_idx = []
        agg_tokens = []
        flag = False

        for i in range(len(tokens)):
            # this token is a part of masked token
            if tokens[i][:2] == '##':
                word_start = word_start + tokens[i][2:]
                flag = True

            else:
                # the aggregate of word just finished
                if flag == True:
                    agg_tokens = agg_tokens[:-1]
                    agg_tokens.append(word_start)
                    # aggregated word have added to list
                    flag = False

                    # current token is a start of new word
                    agg_tokens_idx.append(i)
                    agg_tokens.append(tokens[i])
                    word_start = tokens[i]
                else:
                    # Not masked word, add to list
                    agg_tokens_idx.append(i)
                    agg_tokens.append(tokens[i])
                    word_start = tokens[i]
        
        # The last word is masked word, and waited in word_start
        # remove the added token of the first piece, and add the whole word
        if flag == True:
            agg_tokens = agg_tokens[:-1]
            agg_tokens.append(word_start)

        # Exclude the [CLS] and the [SEP]
        agg_tokens_idx, agg_tokens = agg_tokens_idx[1:-1], agg_tokens[1:-1]

        return agg_tokens_idx, agg_tokens

                
    def agg_attention(self, instance, percentile):
        # Preprocess the instance
        features = self.preprocess_function(instance)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.mp_model(**features)
            attentions = outputs.attentions

        # Convert attentions to numpy without padding
        inputs = self.tokenizer(
                                instance, 
                                add_special_tokens=True,
                                max_length=self.MAX_LEN,
                                truncation=True,
                                padding=False,
                                return_attention_mask=True,
                                return_tensors='pt'
                            )
        input_ids = inputs.input_ids
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        agg_tokens_idx, agg_tokens = self.agg_subwords(tokens)
        # attention length = 12, with each [1,12,512,512]
        # [batch_size, num_heads, sequence_length, sequence_length]
        last_layer = attentions[-1][0]
        # [CLS] is the [0] token. Slice all 12 heads and target tokens
        cls_attention = last_layer[:, 0, :].cpu().data.numpy()
        cls_attention_used = cls_attention[:,agg_tokens_idx]

        # Calculate the target weight (20% of total absolute weights)
        total_attention = np.sum(sum(cls_attention_used))
        target_attention = percentile * total_attention

        # sum the 12 head's attention to each token, then sort from large to small
        indices = list(range(len(agg_tokens)))
        summed_attention = np.sum(cls_attention_used, axis=0)
        tokens_attention = list(zip(indices, agg_tokens, summed_attention))
        sorted_tokens_attention = sorted(tokens_attention, key=lambda x: x[2], reverse=True)

        cum_token_attention = 0
        select_tokens = []
        select_attention = []
        select_positions = []

        # select tokens untill the cumulative attention weight at least equal to the target attention
        for indice, token, attention in sorted_tokens_attention:
                cum_token_attention += attention
                select_positions.append(indice)
                select_tokens.append(token)
                select_attention.append(attention)
                if cum_token_attention >= target_attention:
                    break

        return agg_tokens, select_attention, select_tokens, select_positions
    

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        length = len(value)
        return tuple(int(value[i:i+length//3], 16) for i in range(0, length, length//3))

    def normalize_weights(self, weights):
        """Normalize weights to range [0,1]"""
        min_weight = min(weights)
        max_weight = max(weights)
        range_weight = max_weight - min_weight
        return [(w - min_weight) / range_weight for w in weights]

    def log_normalize_weights(self, weights):
        """Logarithmically normalize weights to range [0,1]"""
        # Apply logarithm (with a small constant to prevent log(0))
        weights = np.abs(weights)
        log_weights = [math.log(w + 0.0001) for w in weights]
        
        min_weight = min(log_weights)
        max_weight = max(log_weights)
        range_weight = max_weight - min_weight

        # Check if all weights are the same (i.e., range is 0)
        if range_weight == 0:
            return [0.5 for _ in log_weights]  # Return 0.5 (midpoint) for all weights
        
        return [(w - min_weight) / range_weight for w in log_weights]
    
    
    def visualize_attention(self, original_text, tokens, weights, idx):
        # Replace newlines with a unique placeholder.
        original_text = original_text.replace('\n', '###NEWLINE###')
        
        # This is the final result text with the highlighted portions.
        result_text = ""

        # Normalize the weights for highlighting intensity.
        normalized_weights = self.log_normalize_weights(weights)

        # Use a pointer to loop over the original text.
        current_position = 0

        for i, token in enumerate(tokens):
            # Case-insensitive search to find the token in the original text.
            start_position = original_text.lower().find(token.lower(), current_position)

            # Append any text before the current token from the original text.
            result_text += original_text[current_position:start_position]

            # If this token needs to be highlighted, wrap it with the appropriate style.
            if i in idx:
                color = "#CC00CC"  

                r, g, b = self.hex_to_rgb(color)
                alpha = normalized_weights[idx.index(i)]
                # Use token from the original text to maintain its capitalization).
                token_to_add = original_text[start_position:start_position + len(token)]
                result_text += f'<span style="background-color: rgba({r}, {g}, {b}, {alpha})">{token_to_add}</span>'
            else:
                result_text += original_text[start_position:start_position + len(token)]

            # Move the pointer past this token for the next iteration.
            current_position = start_position + len(token)

        # Append any remaining text after the last token.
        result_text += original_text[current_position:]

        # Replace the placeholder with <br> for proper line breaks in HTML.
        result_text = result_text.replace('###NEWLINE###', '<br>')
        
        return result_text


    def cls_embed(self, instance):
        # Preprocess the instance
        features = self.preprocess_function(instance)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.mp_model(**features)
            # Ensure that output_hidden_states=True in your model config
            hidden_states = outputs.hidden_states
            # Get the last hidden layer
            last_hidden_state = hidden_states[-1]
            # Extract the [CLS] embedding
            cls_embedding = last_hidden_state[0][0].cpu().numpy()

        return cls_embedding
    

    def cls_mean_pooling(self, instance):
        # Preprocess the instance
        features = self.preprocess_function(instance)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.mp_model(**features)
            # Ensure that output_hidden_states=True in your model config
            hidden_states = outputs.hidden_states
            # Get the last hidden layer
            last_hidden_state = hidden_states[-1]
            # Apply attention mask to ignore padding tokens
            attention_mask = features['attention_mask']
            # [1, 512] to [1, 512, 1] to [1, 512, 768]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            # Remove [CLS] and [SEP] since they containa aggregate info from tokens
            # Remove padded tokens, which has attention_mask = 0
            sum_hidden_state = torch.sum(last_hidden_state[:, 1:-1, :] * mask_expanded[:, 1:-1, :], 1)
            
            mean_pooled = sum_hidden_state / mask_expanded.sum(1)
            mean_pooled = mean_pooled.cpu().numpy()

        return mean_pooled

    def alternate_merge(self, list1, list2):
            result = []
            for i in range(max(len(list1), len(list2))):
                if i < len(list1):
                    result.append(list1[i])
                if i < len(list2):
                    result.append(list2[i])
            return result

    def convert_label(self, num_label):
        return 'survived' if num_label == 0 else 'mortality'
    
    def knn_prompt(self, train_embeddings, train_admission, instance, target_label, metric):
        embed_space = train_embeddings
        embed_instance = [self.cls_embed(instance)]

        # Fit the Nearest Neighbors Model
        nn_model = NearestNeighbors(metric=metric)
        nn_model.fit(embed_space)

        # Initialize parameters
        same_label = []
        opposite_label = []
        retrieved = 0
        batch_size = 10

        # Loop until 3 same label and 3 opposite label retrieved
        while len(same_label) < 3 or len(opposite_label) < 3:
            n_neighbors = retrieved + batch_size
            distances, indices = nn_model.kneighbors(embed_instance, n_neighbors=n_neighbors)

            # Check whether retrieved results fulfill the desired label
            for i in range(retrieved, n_neighbors):
                train_idx = indices[0][i]
                retrieved_label = train_admission.hospital_expire_flag[train_idx]
                if retrieved_label == target_label and len(same_label) < 3:
                    same_label.append(train_idx)
                elif retrieved_label != target_label and len(opposite_label) < 3:
                    opposite_label.append(train_idx)

                # Break early if we have found 3 neighbors of each label
                if len(same_label) >= 3 and len(opposite_label) >= 3:
                    break
            retrieved = n_neighbors

        # Merging the lists alternatively
        merged_indexes = self.alternate_merge(opposite_label, same_label)

        # Initializing a list to store the dictionaries
        prompt_list = []

        # Query the dataframe with the merged_indexes and create the dictionary
        for i in merged_indexes:
            # Convert numerical label to descriptive label
            descriptive_label = self.convert_label(train_admission.hospital_expire_flag[i])
            prompt_list.append({'text': train_admission.text[i], 'label': descriptive_label})

        return merged_indexes, prompt_list