import transformers
from datasets import load_dataset


class TestSampleLength:
    def __init__(self, model_path: str, dataset: str) -> None:
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.dataset = load_dataset(dataset)
   
    def test_sample_average_tokens(self, samples_num : int) -> int:
        total_tokens = 0
        count = 0
        for sample in self.dataset:
            inputs = self.tokenizer(sample, return_tensors='pt')
            
            print(f"the token length of teh inputs is {len(inputs['input_ids'][0])}")
            count += 1
            if count >samples_num:
                break
        return total_tokens
    def test_data():
        print(self.dataset['train'][0:5])
if __name__ == '__main__':
    test_sample_len = TestSampleLength("/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct", "HuggingFaceH4/MATH-500")
    # test_sample_len.test_sample_average_tokens(samples_num=5)
    # print(f"the average token length of the samples is {total_tokens /samples_num}")



