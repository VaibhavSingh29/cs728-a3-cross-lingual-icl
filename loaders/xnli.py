import random
import os
import json

class FewShotXNLI():
    
    def __init__(self, data_dir='data/XNLI-1.0', language_pair='en-en', num_examples=0, seed=42) -> None:
        super().__init__()
        
        self.task_name = 'xnli'
        self.label_list = ["contradiction", "entailment", "neutral"]
        self.label2idx = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.verbalizer = {"no": "contradiction", "yes": "entailment", "also": "neutral"}
        self.lang_abbr = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
        self.template_map = {"en": ["Question", "Answer"], "fr": ["Question", "Réponse"], "ru": ["Вопрос", "Ответ"]}
        self.fewshot_lang = language_pair.split('-')[0]
        self.infer_lang = language_pair.split('-')[1]
        self.seed = seed
        self.data_dir = data_dir
        
        self.num_examples = num_examples
        assert self.num_examples >= 0, "Few-shot examples cannot be less than zero"
        
    def get_labels(self) -> list:
        return self.label_list
    
    def get_verbalizer(self) -> dict:
        return self.verbalizer
    
    def parse_xnli(self, split=None, language=None) -> list:
        assert split in ['dev', 'test'], "Specify one of 'dev' or 'test'"
        assert language in self.lang_abbr, 'Specify a valid language identifier'
        
        file_path = os.path.join(self.data_dir, f"xnli.{split}.jsonl")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for json_line in file:
                line_data = json.loads(json_line)
                if line_data['language'] == language:
                    data.append(line_data)
        return data
    
    def fit_template(self, data=None, fewshot=False) -> list:
        templatized_data = []
        if fewshot:
            for item in data:
                templatized_data.append(f"<s>{item['sentence1']}</s></s>{self.template_map[self.fewshot_lang][0]}: {item['sentence2']}? {self.template_map[self.fewshot_lang][1]}: {item['gold_label']}</s>")
        else:
            for item in data:
                templatized_data.append(f"<s>{item['sentence1']}</s></s>{self.template_map[self.infer_lang][0]}: {item['sentence2']}? {self.template_map[self.infer_lang][1]}:")
        return templatized_data
    def generate_random(self) -> list:
        random.seed(self.seed)
        data = self.parse_xnli(split='dev', language=self.fewshot_lang)
        fewshots = random.sample(data, self.num_examples)
        fewshots = self.fit_template(data=fewshots, fewshot=True)
        return fewshots
    
    def generate_inference_instances(self, choice='random') -> list:
        assert choice in ['random'], "Select a valid sampling method for choosing fewshots"
        test_data = self.parse_xnli(split='test', language=self.infer_lang)[:200]
        templatized_data = self.fit_template(data=test_data)
        instruction = f"<s>Given a premise and a question, choose an answer from one of: 'entailment', 'contradiction' or 'neutral'.</s>"
        compiled_test_data = []
        if choice == 'random':
            fewshots = self.generate_random()
            for item in templatized_data:
                compiled_test_data.append(f"{instruction}\n{'\n'.join(fewshots)}\n{item}")
        return compiled_test_data
    