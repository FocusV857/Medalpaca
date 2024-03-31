from transformers import pipeline
import pandas as pd
import re
from tqdm import tqdm
pl = pipeline("text-generation", model="medalpaca/medalpaca-13b", tokenizer="medalpaca/medalpaca-13b")


op_num = 4
df = pd.read_json(f'../medbullets_op{op_num}.json')
outputs = []
predicted = []
correct = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    question = row['question']
    opa = row['opa']
    opb = row['opb']
    opc = row['opc']
    opd = row['opd']
    if op_num==5:
        ope = row['ope']
        Input = f"The following is a medical multiple choices question. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd} (E) {ope}\nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
    else:
        Input = f"The following is a medical multiple choices question. \nQuestion: {question} (A) {opa} (B) {opb} (C) {opc} (D) {opd}\nPlease choose an answer, strictly following the output format 'Answer:(fill in the letter of the answer)'"
    answer = pl(f"<USER>:{Input} <ASSISTANT>:", max_new_tokens = 1024)
    generated_text = answer[0]['generated_text']
    output = generated_text.replace(f"<USER>:{Input} <ASSISTANT>:", '')
    outputs.append(output)

    pred_idx = re.sub("Answer", "", output)
    pred_idx = re.sub("answer", "", pred_idx)
    pred_idx = re.sub("[^A-E]+", "", pred_idx)
    if pred_idx!='':
        pred_idx = pred_idx[0]
    predicted.append(pred_idx)
    if pred_idx == row['answer_idx']:
        correct +=1
print(f'Acc is {correct/len(df)}')
df['output'] = outputs
df['prediction'] = predicted
df.to_json(f'medalpaca_medbullets_op{op_num}.json')
df.to_csv(f'medalpaca_medbullets_op{op_num}.csv',index=False)