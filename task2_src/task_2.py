import difflib
import unicodedata
import string
import csv

def printshare(msg,  logfile, mode="a"):
    print(msg)

    with open(logfile, mode=mode, encoding='utf-8') as f:
        print(msg, file=f)


def clean_text(text):
    """
    1. Converts text to lowercase.
    2. Replaces non-breaking spaces and other whitespaces with a regular space.
    3. Removes punctuation using unicodedata.
    """
    if not text:
        return ""
    
    # 1. 
    text = text.lower()
    
    # 2.
    text = " ".join(text.split())

    # 3. 
    cleaned_chars = []
    for char in text:
        if not unicodedata.category(char).startswith('P'):
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(' ')
            
    text = "".join(cleaned_chars)
    
    # ensuring space absence for safety
    return " ".join(text.split())

def calculate_metrics(reference, hypothesis, mode='word'):
    """
    Compares two texts and calculates S, I, D.
    mode='word' -> WER (Word Error Rate)
    mode='char' -> CER (Character Error Rate)
    """
    
    if mode == 'word':
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
    else: # mode == 'char'
        # CER => replace spaces with underscores for stability
        ref_tokens = list(reference.replace(' ', '_'))
        hyp_tokens = list(hypothesis.replace(' ', '_'))

    # difflib to find differences. autojunk disabled bcs CER suffer otherwise.
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens, autojunk=False)
    
    S, I, D, C = 0, 0, 0, 0
    alignment_table = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_segment = ref_tokens[i1:i2]
        hyp_segment = hyp_tokens[j1:j2]
        
        if tag == 'equal':
            # Correct matches
            for r, h in zip(ref_segment, hyp_segment):
                alignment_table.append((r, "*", "OK"))
                C += 1
                
        elif tag == 'replace':
            # Substitution
            len_ref = len(ref_segment)
            len_hyp = len(hyp_segment)
            min_len = min(len_ref, len_hyp)

            for k in range(min_len):
                alignment_table.append((ref_segment[k], hyp_segment[k], "S"))
                S += 1
            
            # lengths differ => record the rest as Deletions or Insertions
            if len_ref > len_hyp:
                for k in range(min_len, len_ref):
                    alignment_table.append((ref_segment[k], '""', "D"))
                    D += 1
            elif len_hyp > len_ref:
                for k in range(min_len, len_hyp):
                    alignment_table.append(('""', hyp_segment[k], "I"))
                    I += 1

        elif tag == 'delete':
            # Deletion
            for word in ref_segment:
                alignment_table.append((word, '""', "D"))
                D += 1
                
        elif tag == 'insert':
            # Insertion
            for word in hyp_segment:
                alignment_table.append(('""', word, "I"))
                I += 1

    N = len(ref_tokens)
    error_rate = (S + D + I) / N if N > 0 else -1.0
    
    return {
        "WER/CER": error_rate,
        "S": S,
        "I": I,
        "D": D,
        "N": N,
        "Alignment": alignment_table
    }

def print_results(title, result, file):
    printshare(f"\n--- {title} ---", file, mode="w")
    printshare(f"S (Substitutions): {result['S']}", file)
    printshare(f"I (Insertions): {result['I']}", file)
    printshare(f"D (Deletions): {result['D']}", file)
    printshare(f"N (Total tokens in reference): {result['N']}", file)
    printshare(f"**Error Rate: {result['WER/CER']:.2%}**", file)
    
    printshare("\nAlignment Table:", file)
    printshare(f"{'REFERENCE':<20} | {'HYPOTHESIS':<20} | {'TYPE':<5}", file)
    printshare("-" * 50, file)
    for ref, hyp, error_type in result['Alignment']:
        #show '*' for OK, or the actual hypothesis
        hyp_display = hyp if hyp != "*" else "*"     
        printshare(f"{str(ref):<20} | {str(hyp_display):<20} | {error_type}", file)

def print_results_csv(result, file):
    """
    Saves the alignment table to a CSV file without decorative headers.
    Columns: Reference, Hypothesis, Type
    """
    with open(file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['Reference', 'Hypothesis', 'Type'])
        
        for ref, hyp, error_type in result['Alignment']:
            # show '*' for OK, or the actual hypothesis
            hyp_display = hyp if hyp != "*" else "*"
            writer.writerow([ref, hyp_display, error_type])

if __name__ == "__main__":
    reference_raw = """І раптом літо, згадавши, як добре йому було, прокидається з невчасно
    розпочатої сплячки, розливає сонце над нивами та лісами. І спалахує навколо
    бабине літо! Бузкова герань, гнучкі петрові батоги, жовті кульбабки, білосніжні
    ромашки – усе знову квітне, сподіваючись на тепло. Удруге вбирається у
    весільний одяг глуха кропива, червоніють навіть духмяні сунички. Небеса,
    осяяні яскравим холодним промінням, низько схиляються до справжніх осінніх
    квітів: жоржин, айстр, чорнобривців, сальвій. Вони будуть стояти до
    останнього, закриваючись пелюстками від перших, ще не дуже жорстоких
    морозів. Але це буде за кілька днів, всього кілька днів, а зараз –
    бабине літо! Знову замайоріли в повітрі метелики, знову гудуть джмелі, знову
    снують до квіток ніби і не стомлені працею бджоли."""
    wav2vec2_hypothesis_raw = """і раптом літо згадавши як добре йому було прокидаються з невчасно
    розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо 
    баби не літо бускова гирань гнучки петрові батоги жовті кульбабки білосніжні
    ромажки усе знову квітне сподіваючись на тепло удруге вбираються у весільний
    одяг глуха кропила червоніють навіть духмяні сунички небеса осяєні є скравим
    холодним промінням низько схиляються до справжніх осінніх квітів жоржин айстр
    чорнобриєвців сальвій вони будуть стояти до останнього закриваючись палюстками
    від перших ще недуже жорстоких морозів але це буде за кілька днів всього 
    кілька днів а зараз баби не літо знову замайоріли в повітрі метелики знову
    годуть джмелі знову снують доквіток ніби і нестомлені праце йобджоли"""

    dswlm_hypothesis_raw = """і раптом літо згадавши як добре йому було прокидається з невчасно 
    розпочати сплячки розливає сонце санаторного квітникарство канонізованого
      основоположника аортокоронарного """
    
    wav2vec2_anton_hypothesis_raw="""і раптом літо згадавши як добре йому було прокидаються з невчасно
      розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо баби не
        літо бускова гирань гнучки петрові батоги жовті кульбабки білосніжні ромажки усе 
        знову квітне сподіваючись на тепло удруге вбираються у весільний одяг глуха кропива
          червоніють навіть духмяні сунечки небиса осяяні яскравим холодним промінням низько
            схиляються до справжніх осінніх квітів жоржин айстир чорнобривців сальвій вони будуть
              стояти до останнього закриваючись пелюстками від перших ще не дуже жорстоких морозів
                але це буде за кілька днів всього кілька днів а зараз баби не літо знову замайоріли
                  в повітрі метелики знову гудуть джмелі знову снують доквіток ніби і нестомлені
                    працею бджоли"""

    wav2vec2_vika_hypothesis_raw="""і раптом літо згадавши як добре йому було прокидаються з невчасно
      розпочатої сплячки розливає сонце над нивами та лісами і спалахує навколо баби не літо бускова
        гирань гнучкі петрові батоги жовці кульбабки білосніжні ромашки усе знову квітне сподіваючись
          на тепло удруге вбирається у весільний одяг глуха кропива червоніють навіть духм'яні сунечки
            небиса усяєні яскравим холодним промінням низько схиляються до справжніх осінніх квітів
              жоржен айстер чорнобривців сальвій вони будуть стояти до останнього закриваючись пелюстками
                від перших ще не дуже жорстоких морозів але це буде за кілька днів всього кілька днів
                  а зараз баби не літо знову замайоріли в повітрі метелики знову гудуть джмелі знову 
                  снують доквіто кніби і нестомлені працею бджоли"""
    print("raw ref:", reference_raw)
    print("raw hyp: ", wav2vec2_vika_hypothesis_raw) #wav2vec2_hypothesis_raw

    ref_clean = clean_text(reference_raw)
    hyp_clean = clean_text(wav2vec2_vika_hypothesis_raw) #wav2vec2_hypothesis_raw

    print(f"\nclean ref: '{ref_clean}'")
    print(f"clean hyp: '{hyp_clean}'")

    wer_result = calculate_metrics(ref_clean, hyp_clean, mode='word')
    print_results("Wav2Vec2 WER", wer_result, file="wav2vec2_vika_visual_wer.txt")
    print_results_csv(wer_result, file="wav2vec2_vika_wer.csv")

    cer_result = calculate_metrics(ref_clean, hyp_clean, mode='char')
    print_results("Wav2Vec2 CER", cer_result, file="wav2vec2_vika_visual_cer.txt")
    print_results_csv(cer_result, file="wav2vec2_vika_cer.csv")