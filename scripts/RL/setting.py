from transformers import LineByLineTextDataset
import re
import string

ROOT = '/Users/jliu/PycharmProjects/CF_RL'


def tokenize_data(tokenizer,data_path,block_size:int):
    """tokenize the dataset"""

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_path,
        block_size=block_size,
    )
    return dataset


def remove_annotation(sentences):
    return [re.sub(r'\s*\(.*?\)\s*|\s*\*.*?\*\s*', ' ', sentence).strip() for sentence in sentences]

def remove_emoji(text):
    # Regex pattern to match emojis and remove them
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"  # Emoticons
                                "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                                "\U0001F700-\U0001F77F"  # Alchemical Symbols
                                "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                "\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
                                "\U0001FA00-\U0001FA6F"  # Chess Symbols
                                "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
                                "\U00002702-\U000027B0"  # Dingbats
                                "\U000024C2-\U0001F251"  # Enclosed Characters
                                "]", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

def preprocess(sent: str, remove_punc=False) -> str:
    try:
        # Remove other annotations and convert to lowercase
        sent = remove_annotation([sent])[0].lower()
        # Remove emojis
        sent = remove_emoji(sent)
        # Remove <PAUSE> and <SILENCE> tags, along with their annotations and extra spaces
        text = re.sub(r'\s*<[^>]*>\s*', ' ', sent).strip()
        # Optionally remove punctuation
        if remove_punc:
            text = text.translate(str.maketrans('', '', string.punctuation))
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        text = sent
    return text