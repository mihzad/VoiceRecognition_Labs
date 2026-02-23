import tokenize_uk
import torch

def get_word_predictions(model, tokenizer, texts, is_split_to_words=False, device='cpu'):
    words_res = []
    y_res = []

    if not is_split_to_words:
        texts = [tokenize_uk.tokenize_words(text) for text in texts]

    for text in texts:
        size = len(text)
        idx_list = [idx + 1 for idx, val in enumerate(text) if val in ['.', '?', '!']]
        if len(idx_list):
            sents = [text[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
        else:
            sents = [text]

        y_res_x = []
        words_res_x = []
        for sent_tokens in sents:
            tokenized_inputs = [101]
            word_ids = [None]
            for word_id, word in enumerate(sent_tokens):
                word_tokens = tokenizer.encode(word)[1:-1]
                tokenized_inputs += word_tokens
                word_ids += [word_id]*len(word_tokens)
            tokenized_inputs = tokenized_inputs[:(tokenizer.model_max_length-1)]
            word_ids = word_ids[:(tokenizer.model_max_length-1)]
            tokenized_inputs += [102]
            word_ids += [None]

            torch_tokenized_inputs = torch.tensor(tokenized_inputs).unsqueeze(0)
            torch_attention_mask = torch.ones(torch_tokenized_inputs.shape)
            predictions = model.forward(input_ids=torch_tokenized_inputs.to(device), attention_mask=torch_attention_mask.to(device))
            predictions = torch.argmax(predictions.logits.squeeze(), axis=1).cpu().numpy()
            predictions = [model.config.id2label[i] for i in predictions]

            previous_word_idx = None
            sent_words = []
            predictions_words = []
            word_tokens = []
            first_pred = None
            for i, word_idx in enumerate(word_ids):
                if word_idx != previous_word_idx:
                    sent_words.append(tokenizer.decode(word_tokens))
                    word_tokens = [tokenized_inputs[i]]
                    predictions_words.append(first_pred)
                    first_pred = predictions[i]
                else:
                    word_tokens.append(tokenized_inputs[i])      
                previous_word_idx = word_idx

            words_res_x.extend(sent_words[1:])
            y_res_x.extend(predictions_words[1:])

        words_res.append(words_res_x)
        y_res.append(y_res_x)

    return words_res, y_res
	
	
def recover_text(text, model, tokenizer, device='cpu'):
    text_words, text_preds = get_word_predictions(model, tokenizer, [text], device=device)
    text_words = text_words[0]
    text_preds = text_preds[0]
    
    for i in range(len(text_words)):
        pred_case = text_preds[i][:2]
        pred_punct = text_preds[i][2:]
        if pred_case=='AA':
            text_words[i] = text_words[i].upper()
        elif pred_case=='Aa':
            text_words[i] = text_words[i].capitalize()
        if pred_punct!='O':
            text_words[i] += pred_punct

    return ' '.join(text_words)