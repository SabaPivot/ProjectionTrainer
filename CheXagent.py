from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-2-3b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/CheXagent-2-3b", trust_remote_code=True)

print(model)