from loaders.xnli import FewShotXNLI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

def set_seed(value=42):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)

def main():
    
    set_seed()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xnli = FewShotXNLI(num_examples=4)
    text = xnli.generate_inference_instances(choice='random')[0]
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    model = AutoModelForCausalLM.from_pretrained("facebook/xglm-564M").to(device)
    
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_new_tokens=5, do_sample=True, top_k=5, top_p=0.95)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()