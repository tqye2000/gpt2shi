################################################################################
# Test GPT-2 Text Generation Model                                             #
# This code loads a pre-trained GPT-2 model and uses it to generate text       #
# based on a given prompt.
#                                                                              #
# When       | Who         | What                                              #
# 20/01/2023 |Tian-Qing Ye | Creation                                          #
################################################################################
from transformers import BertTokenizer, TFGPT2LMHeadModel, TextGenerationPipeline
import logging
import sys
import os
import re

###################################################
# Only displaying the erros, suspend the warnings
###################################################
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

#model_name="uer/gpt2-chinese-ancient"
model_name="uer/gpt2-chinese-poem"
#model_name="uer/gpt2-chinese-lyric"
#model_name="uer/gpt2-chinese-couplet"

###################################################
# Load the tokenizer and pre-trained model
###################################################
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)
#tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
#model = TFGPT2LMHeadModel.from_pretrained(model_name, use_cache=True, local_files_only=True)

OUTPUT_FOLDER = "outputs"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "generated_poems.txt")
   
def postprocess(text):
    #first split into sentences
    sentences = re.split(r'，|！|。', text)
    new_sentences = []
    for sen in sentences:
        sen = sen.replace("[CLS]", "").strip()
        new_sentences.append(sen)

    #now remove sentence with "odd" number of chars
    poem = ""
    poems = []
    skip_next = False
    for text in new_sentences:
        if skip_next == True:
            skip_next = False
            continue

        if(len(text) == 5 or len(text) == 7):
            poem += text + ' '
        else:
            skip_next = True

    #group into jue
    sentences = poem.split(' ')
    sentences = list(filter(None, sentences))

    poems = []
    s = '\n'
    poems.append(s.join(sentences[0:4]))
    poems.append(s.join(sentences[-8:]))
    #poems.append(sentences[0:4])
    #poems.append(sentences[-8:])
    return poems


##############################################
################ MAIN ########################
##############################################
def main(argv):

    #Enter Loop
    while True:
        prompts = []
        user_input = input("Please enter some text (or 'bye' to exit): ")
        if user_input.strip().lower() == "bye":
            break
        else:
            if user_input.strip().lower() == "batch":
                prompts = [
                    "细雨绵绵",
                    "荷塘花下",
                ]
            else:
                if user_input.strip().lower() == "ok" or len(user_input.strip()) < 1:
                    break
                else:
                    print("You entered: ", user_input)
                    print("Responsed:")
                    prompts.append(user_input)
    
        #============ Note ==============================================
        #For [Q]: and [A]: 
        #we will have to follow the explicit format of “[Q]: X, [A]:” before letting it attempt to autocomplete. 
        #================================================================
        #Generating response
        generated_texts = []
        temp = 1.0
        topp = 0.5
        topk = 30
        nbeams = 4
        nreturns = 1
        f = open(OUTPUT_FILE, "a", encoding='utf-8',)
        for prompt in prompts:
            prompt = f"[CLS]{prompt}"
            text_generator = TextGenerationPipeline(model, tokenizer)   
            outputs = text_generator(prompt, 
                                     max_length=129, 
                                     num_beams=nbeams,
                                     no_repeat_ngram_size=2,
                                     do_sample=True,
                                     top_k=topk,
                                     num_return_sequences=nreturns,
                                     early_stopping=False)
            for i in range(nreturns):
                text = outputs[i]['generated_text']
                print(f'Generated{i+1}: {text}')
                text = text.replace(" ", "").strip()
                f.write(f'\nGenerated{i+1}: {text}\n\n')
                poems = postprocess(text)
                f.write(f'Processed{i+1}: {poems}')
                print (poems)

            print(100 * '-' + '\n')
        
        f.close()


##############################
if __name__ == "__main__":
    main(sys.argv)

