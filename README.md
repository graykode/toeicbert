## TOEIC-BERT

### 76% Correct rate with ONLY Pre-Trained BERT model in TOEIC!!



This is project as topic: `TOEIC(Test of English for International Communication) problem solving using pytorch-pretrained-BERT model.` The reason why I used huggingface's [pytorch-pretrained-BERT model](<https://github.com/huggingface/pytorch-pretrained-BERT>) is for pre-training or to do fine-tune more easily.  **I've solved the only blank problem, not the whole problem.** There are two types of blank issues:

1. Selecting Correct Grammar Type.

```
Q) The music teacher had me _ scales several times.
  1. play (Answer)
  2. to play
  3. played
  4. playing
```

2. Selecting Correct Vocabulary Type.

```
Q) The wet weather _ her from going playing tennis.
  1. interrupted
  2. obstructed
  3. impeded
  4. discouraged (Answer)
```



#### BERT Testing

1. input

```json
{
    "1" : {
        "question" : "Business experts predict that the upward trend is _ to continue until the end of next year.",
        "answer" : "likely",
        "1" : "potential",
        "2" : "likely",
        "3" : "safety",
        "4" : "seemed"
    }
}
```

2. output

```
=============================
Question : Business experts predict that the upward trend is _ to continue until the end of next year.

Real Answer : likely

1) potential 2) likely 3) safety 4) seemed

BERT's Answer => [likely]
```



#### Why BERT?

In pretrained BERT, It contains contextual information. So It can find more contextual or grammatical sentences, not clear, a little bit. I was inspired by grammar checker from [blog post](<https://www.scribendi.ai/can-we-use-bert-as-a-language-model-to-assign-score-of-a-sentence/>).

> [Can We Use BERT as a Language Model to Assign a Score to a Sentence?](<https://www.scribendi.ai/can-we-use-bert-as-a-language-model-to-assign-score-of-a-sentence/>)
>
> BERT uses a bidirectional encoder to encapsulate a sentence from left to right and from right to left. Thus, it learns two representations of each word-one from left to right and one from right to left-and then concatenates them for many downstream tasks.



## Evaluation

<p align="center"><img width="500" src="https://raw.githubusercontent.com/graykode/toeicbert/master/images/baseline.gif" /></p>

I had evaluated with only **pretrained BERT model(not fine-tuning)** to check grammatical or lexical error. Above mathematical expression, `X` is a question sentence. and `n` is number of questions : `{a, b, c, d}`. `C` subset means answer candidate tokens : `C` of `warranty` is `['warrant', '##y']`. `V` means total Vocabulary.

There's a problem with more than one token. I solved this problem by getting the average value of each tensor. ex) `is being formed` as `['is', 'being', 'formed']` 

Then, we find argmax in `L_n(T_n)`.



<p align="center"><img width="350" src="https://raw.githubusercontent.com/graykode/toeicbert/master/images/prediction.gif" /></p>

```python
predictions = model(question_tensors, segment_tensors)

# predictions : [batch_size, sequence_length, vocab_size]
predictions_candidates = predictions[0, masked_index, candidate_ids].mean()
```



#### Result of Evaluation.

Fantastic result with **only pretrained BERT model**

- `bert-base-uncased`: 12-layer, 768-hidden, 12-heads, 110M parameters
- `bert-large-uncased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
- `bert-base-cased`: 12-layer, 768-hidden, 12-heads , 110M parameters
- `bert-large-cased`: 24-layer, 1024-hidden, 16-heads, 340M parameters

Total 7067 datasets: make non-deterministic with `model.eval()`

|             | bert-base-uncased | bert-base-cased | bert-large-uncased | bert-large-cased |
| :---------: | :---------------: | :-------------: | :----------------: | :--------------: |
| Correct Num |       5192        |      5398       |        5321        |       5148       |
|   Percent   |      73.46%       |     76.38%      |       75.29%       |      72.84%      |



## Quick Start with Python pip Package.

**Start with pip**

```shell
$ pip install toeicbert
```



**Run & Option**

```shell
$ python -m toeicbert --model bert-base-uncased --file test.json
```

- `-m, --model` : bert-model name in huggingface's pytorch-pretrained-BERT : `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`, `bert-large-cased`.

- `-f, --file` : json file to evalution, see json format, [test.json](test.json). 

  **key(question, 1, 2, 3, 4)  is required options, but answer not.**

  `_` in question will be replaced to `[MASK]`

```json
{
    "1" : {
        "question" : "The music teacher had me _ scales several times.",
        "answer" : "play",
        "1" : "play",
        "2" : "to play",
        "3" : "played",
        "4" : "playing"
    },
    "2" : {
        "question" : "The music teacher had me _ scales several times.",
        "1" : "play",
        "2" : "to play",
        "3" : "played",
        "4" : "playing"
    }
}
```



## Author

- Tae Hwan Jung(Jeff Jung) @graykode, Kyung Hee Univ CE(Undergraduate).
- Author Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)

Thanks for Hwan Suk Gang(Kyung Hee Univ.) for collecting Dataset(`7114` datasets)