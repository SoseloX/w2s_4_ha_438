import regex as re


def remove_substring(text, substring):
    return text.replace(substring, "")


def normalize_text(text):
    text = remove_substring(text, "The answer is")
    # Convert to lowercase to make the comparison case-insensitive
    text = text.lower()
    # Remove articles: 'a', 'an', 'the'
    text = re.sub(r'\b(a|an|the)\b', '', text)
    # Normalize whitespace, collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[\p{P}\p{S}]', '', text)
    # Normalize numeric expressions (e.g., removing commas in numbers)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Strip leading/trailing whitespace that might be left after other replacements
    text = text.strip()
    return text


def exact_match_score(prediction, ground_truth):
    return (normalize_text(prediction) == normalize_text(ground_truth))






def list_exact_match_score(prediction, ground_truth):
    return any(normalize_text(prediction) == normalize_text(gt) for gt in ground_truth)




def mutual_acc(prediction, ground_truth):
    return (normalize_text(prediction) in normalize_text(ground_truth)) or (normalize_text(ground_truth) == normalize_text(prediction))


def list_acc(prediction, ground_truth):
    return any(normalize_text(prediction) in normalize_text(gt) for gt in ground_truth)


def list_mutual_acc(prediction, ground_truth):
    return any(normalize_text(prediction) in normalize_text(gt) for gt in ground_truth) or any(normalize_text(gt) in normalize_text(prediction) for gt in ground_truth)



def majority_vote(prediction, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    gt_dict = {}
    gt_dict[ground_truth[0]] = 0
    for pred in prediction:
        if pred in ground_truth:
            gt_dict[ground_truth[0]] = gt_dict.get(ground_truth[0], 0) + 1
        else:
            gt_dict[pred] = gt_dict.get(pred, 0) + 1

    max_value = max(gt_dict.values())
    if gt_dict[ground_truth[0]] == max_value:
        return True
    else:
        return False

