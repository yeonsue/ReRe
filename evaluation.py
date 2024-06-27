import evaluate
import json
from evaluate import load
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

ACC=0
total_epoch=30
for iter in range(total_epoch):
    
    nlx_result_path='results/unf_captions_full_{}.json'.format(iter)
    
    with open(nlx_result_path, 'r') as i:
        result_json = json.load(i)
        
    with open('nle_data/VQA-X/vqaX_test.json', 'r') as j:
        test_json = json.load(j)

    with open('nle_data/VQA-X/cocoEval_test.json', 'r') as k:
        unf_references = json.load(k)

    # seperate the 'result_json' to answer and explanation
    # predicted_caption and reference_caption's fomat should be match with below format
    # predictions = {image_id: [predicted_caption1, predicted_caption2, ...]}
    # references = {image_id: [reference_caption1, reference_caption2, ...]}

    error=0
    result_answer=[]
    unf_predictions={}
    for result in result_json:
        image_ids = result["image_id"]
        sentence=result['caption']
        if 'because' in sentence:
            words=sentence.split(' because ',1)
            result_answer.append(words[0])
            unf_predictions[str(image_ids)]=[words[1]]
        else:
            error+=1
            result_answer.append('')
            unf_predictions[str(image_ids)]=['']
    print('number of error output in answer:',error)


    # unf metric score
    print('unf metric score')
    for evaluator in [Cider(),Bleu(),Meteor(),Rouge()]: # Spice()
        score, scores = evaluator.compute_score(unf_references,unf_predictions)
        print(str(evaluator),score)
    
    
    # unf bertscore
    # filter
    f_references,f_predictions = unf_references,unf_predictions
    bertscore = load("bertscore")
    i=0
    score=0
    precision=0
    err = []
    explanations = list(unf_predictions.values())
    for key, value in test_json.items():
        ans_dic={}
        for answers in value['answers']:
            if answers['answer'] in ans_dic:
                ans_dic[answers['answer']]+=1
            else:
                ans_dic[answers['answer']]=1
        # answer = max(ans_dic,key=lambda k:ans_dic[k])

        if result_answer[i] in ans_dic.keys():
            precision+=1
            references=value['explanation']
            prediction=[explanations[i][0] for k in range(len(references))]
            bert_score=bertscore.compute(predictions=prediction, references=references, lang="en")

            # bert_score is measure with average score
            s=0
            for b in bert_score['f1']:
                s+=float(b)
            score+=s/len(bert_score['f1'])
            
            ## perhaps if you want to measure bert_score with Max score
            # score += max(bert_score['precision'])

        else:
            del f_references[key]
            del f_predictions[key]
        i+=1
    total_num=len(result_json)
    print('precision:',precision)
    print("'f1': ",score/precision," 'precision': ",precision/total_num)

    print("number of f_references: ",len(f_references))
    print('filtered metric score')
    for evaluator in [Cider(),Bleu(),Meteor(),Rouge()]: # Spice()
        score, scores = evaluator.compute_score(f_references,f_predictions)
        print(str(evaluator),score)

    if ACC<precision/total_num:
        ACC=precision/total_num
        best_result=iter
    print('best_result iteration is ',best_result)