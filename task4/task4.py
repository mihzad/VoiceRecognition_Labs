import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import tokenize_uk
from uk_puntcase.get_predictions import get_word_predictions, recover_text

def extract_etalon_tags(etalon_text):
    """
    Extracts punctuation tags from the ground truth (etalon) text.
    Uses 'tokenize_uk' to ensure the token splitting exactly matches 
    how the model processes the recognized text.
    """
    tokens = tokenize_uk.tokenize_words(etalon_text)
    tags = []
    
    for i in range(len(tokens)):
        # If the token is a word (not one of the allowed punctuations)
        if tokens[i] not in ['.', ',', '?', '!']:
            # Check if the next token is a punctuation mark
            if i + 1 < len(tokens) and tokens[i+1] in ['.', ',', '?', '!']:
                tags.append(tokens[i+1])
            else:
                tags.append('O') # 'O' means no punctuation
    return tags

def evaluate_punctuation(base_to_recognize, etalon_text):
    print("Loading 'ukr-models/uk-punctcase' model and tokenizer...")
    model_name = "ukr-models/uk-punctcase"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print("Running predictions...")
    _, text_preds = get_word_predictions(model, tokenizer, [base_to_recognize], device=device)
    text_preds = text_preds[0]
    # Extract just the punctuation part from the model's tags (from index 2 onward)
    pred_tags = [tag[2:] for tag in text_preds]
    
    # Extract the ideal tags from the etalon text
    etalon_tags = extract_etalon_tags(etalon_text)
    
    # Sanity check: verify the word counts match
    if len(etalon_tags) != len(pred_tags):
        print("\n[WARNING] Word counts do not match! Alignment may be slightly off.")
        print(f"Etalon words: {len(etalon_tags)} | Predicted words: {len(pred_tags)}")
        # Truncate to the shortest list to prevent index out-of-bounds errors
        min_len = min(len(etalon_tags), len(pred_tags))
        etalon_tags = etalon_tags[:min_len]
        pred_tags = pred_tags[:min_len]

    # Calculate TP, FP, TN, FN
    TP = FP = TN = FN = 0

    for t_true, t_pred in zip(etalon_tags, pred_tags):
        if t_true == t_pred:
            if t_true == 'O':
                TN += 1  # True Negative: Both agree there is NO punctuation
            else:
                TP += 1  # True Positive: Both agree on the EXACT punctuation mark
        else:
            if t_true == 'O' and t_pred != 'O':
                FP += 1  # False Positive: Model hallucinated a punctuation mark
            elif t_true != 'O' and t_pred == 'O':
                FN += 1  # False Negative: Model missed a punctuation mark
            else:
                # Mismatch (e.g., ground truth is '.', but model predicted ',')
                # Counts as 1 False Positive for the ',' and 1 False Negative for the '.'
                FP += 1  
                FN += 1

    # Calculate Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n--- Input Text ---")
    print(base_to_recognize)
    print("\n--- Predicted Punctuation-recovered Text ---")
    # Recover text by inserting predicted punctuation after each word
    tokens = tokenize_uk.tokenize_words(base_to_recognize)
    recovered = []
    for token, tag in zip(tokens, pred_tags):
        result_token = token
        if tag != 'O':
            result_token += tag
        recovered.append(result_token)
    recovered_text = ' '.join(recovered)
    print(recovered_text)
    print("\n--- Etalon Text ---")
    print(etalon_text)

    print("\n--- Confusion Matrix ---")
    print(f"TP (True Positives):  {TP}")
    print(f"FP (False Positives): {FP}")
    print(f"TN (True Negatives):  {TN}")
    print(f"FN (False Negatives): {FN}")
    
    print("\n--- Final Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return precision, recall, f1


if __name__ == "__main__":
    wav2vec2_recognized_me = """і раптом літо згадавши як добре йому було прокидаються з невчасно
    розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо 
    баби не літо бускова гирань гнучки петрові батоги жовті кульбабки білосніжні
    ромажки усе знову квітне сподіваючись на тепло удруге вбираються у весільний
    одяг глуха кропила червоніють навіть духмяні сунички небеса осяєні є скравим
    холодним промінням низько схиляються до справжніх осінніх квітів жоржин айстр
    чорнобриєвців сальвій вони будуть стояти до останнього закриваючись палюстками
    від перших ще недуже жорстоких морозів але це буде за кілька днів всього 
    кілька днів а зараз баби не літо знову замайоріли в повітрі метелики знову
    годуть джмелі знову снують доквіток ніби і нестомлені праце йобджоли"""
    wav2vec2_etalon_me = """і раптом літо, згадавши, як добре йому було, прокидаються з невчасно 
    розпочатої сплячки, розливає сонце над нивами та лісами. і спалахує навколо
      баби не літо. бускова гирань, гнучки петрові батоги, жовті кульбабки, білосніжні
        ромажки, усе знову квітне, сподіваючись на тепло. удруге вбираються у весільний
          одяг глуха кропила, червоніють навіть духмяні сунички. небеса, осяєні є скравим
            холодним промінням, низько схиляються до справжніх осінніх квітів, жоржин, айстр,
              чорнобриєвців, сальвій. вони будуть стояти до останнього, закриваючись палюстками
                від перших, ще недуже жорстоких морозів. але це буде за кілька днів, всього 
                кілька днів, а зараз, баби не літо. знову замайоріли в повітрі метелики,
                  знову годуть джмелі, знову снують доквіток ніби і нестомлені праце йобджоли."""
    
    wav2vec2_recognized_anton = "і раптом літо згадавши як добре йому було прокидаються з невчасно розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо баби не літо бускова гирань гнучки петрові батоги жовті кульбабки білосніжні ромажки усе знову квітне сподіваючись на тепло удруге вбираються у весільний одяг глуха кропива червоніють навіть духмяні сунечки небиса осяяні яскравим холодним промінням низько схиляються до справжніх осінніх квітів жоржин айстир чорнобривців сальвій вони будуть стояти до останнього закриваючись пелюстками від перших ще не дуже жорстоких морозів але це буде за кілька днів всього кілька днів а зараз баби не літо знову замайоріли в повітрі метелики знову гудуть джмелі знову снують доквіток ніби і нестомлені працею бджоли"
    wav2vec2_etalon_anton = "і раптом літо, згадавши, як добре йому було, прокидаються з невчасно розпочатої сплячки, розливає сонце над нивами та лісами. і спалахує навколо баби не літо. бускова гирань, гнучки петрові батоги, жовті кульбабки, білосніжні ромажки, усе знову квітне, сподіваючись на тепло. удруге вбираються у весільний одяг глуха кропива, червоніють навіть духмяні сунечки. небиса, осяяні яскравим холодним промінням, низько схиляються до справжніх осінніх квітів, жоржин, айстир, чорнобривців, сальвій. вони будуть стояти до останнього, закриваючись пелюстками від перших, ще не дуже жорстоких морозів. але це буде за кілька днів, всього кілька днів, а зараз, баби не літо. знову замайоріли в повітрі метелики, знову гудуть джмелі, знову снують доквіток ніби і нестомлені працею бджоли."
    
    wav2vec2_recognized_vika = "і раптом літо згадавши як добре йому було прокидаються з невчасно розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо баби не літо бускова гирань гнучкі петрові батоги жовці кульбабки білосніжні ромашки усе знову квітне сподіваючись на тепло удруге вбирається у весільний одяг глуха кропива червоніють навіть духм'яні сунечки небиса усяєні яскравим холодним промінням низько схиляються до справжніх осінніх квітів жоржен айстер чорнобривців сальвій вони будуть стояти до останнього закриваючись пелюстками від перших ще не дуже жорстоких морозів але це буде за кілька днів всього кілька днів а зараз баби не літо знову замайоріли в повітрі метелики знову гудуть джмелі знову снують доквіто кніби і нестомлені працею бджоли"
    wav2vec2_etalon_vika = "і раптом літо, згадавши, як добре йому було, прокидаються з невчасно розпочатої сплячки, розливає сонце над нивами та лісами. і спалахує навколо баби не літо. бускова гирань, гнучкі петрові батоги, жовці кульбабки, білосніжні ромашки, усе знову квітне, сподіваючись на тепло. удруге вбирається у весільний одяг глуха кропива, червоніють навіть духм'яні сунечки. небиса, усяєні яскравим холодним промінням, низько схиляються до справжніх осінніх квітів, жоржен, айстер, чорнобривців, сальвій. вони будуть стояти до останнього, закриваючись пелюстками від перших, ще не дуже жорстоких морозів. але це буде за кілька днів, всього кілька днів, а зараз, баби не літо. знову замайоріли в повітрі метелики, знову гудуть джмелі, знову снують доквіто кніби і нестомлені працею бджоли."

    #evaluate_punctuation(wav2vec2_recognized_me, wav2vec2_etalon_me)
    #evaluate_punctuation(wav2vec2_recognized_anton, wav2vec2_etalon_anton)
    evaluate_punctuation(wav2vec2_recognized_vika, wav2vec2_etalon_vika)