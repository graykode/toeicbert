"""
    reference : https://github.com/huggingface/pytorch-pretrained-BERT/issues/80,
                https://www.scribendi.ai/can-we-use-bert-as-a-language-model-to-assign-score-of-a-sentence/
    code by Tae Hwan Jung(@graykode)
"""
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import torch
import json

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

bertmodel = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(bertmodel)

model = BertForMaskedLM.from_pretrained(bertmodel).to(device)
model.eval()

def get_score(question_tensors, segment_tensors, masked_index, candidate):

    candidate_tokens = tokenizer.tokenize(candidate) # warranty -> ['warrant', '##y']
    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)

    predictions = model(question_tensors, segment_tensors)
    predictions_candidates = predictions[0, masked_index, candidate_ids].mean()

    return predictions_candidates.item()

"""
    json file format
    {
        "1" : {
            "question" : "The teacher had me _ scales several times a day.",
            "answer" : "play",
            "1" : "play",
            "2" : "to play",
            "3" : "played",
            "4" : "playing"
        },
        "2" : {
    
        }
    }
"""

with open('data.json') as data_file:
    data = json.load(data_file)

correct = 0

if __name__ == "__main__":
    for (k, row) in data.items():
        question_tokens = tokenizer.tokenize(row['question'])
        masked_index = question_tokens.index('[MASK]')

        # make segment which is divided with sentence A or B, but we set all '0' as sentence A
        segment_ids = [0] * len(question_tokens)
        segment_tensors = torch.tensor([segment_ids]).to(device)

        # question tokens convert to ids and tensors
        question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        question_tensors = torch.tensor([question_ids]).to(device)

        candidates = [row['1'], row['2'], row['3'], row['4']]
        predict_tensor = torch.tensor([get_score(question_tensors, segment_tensors,
                                        masked_index , candidate) for candidate in candidates])
        predict_idx = torch.argmax(predict_tensor).item()

        if row['answer'] == candidates[predict_idx]:
            correct += 1
        print('%s/%d : %d => %s =? %s' % (k, len(data), correct, row['answer'], candidates[predict_idx]))
