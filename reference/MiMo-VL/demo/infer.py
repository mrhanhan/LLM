# modified from https://github.com/ByteDance-Seed/Seed1.5-VL/blob/main/GradioDemo/infer.py
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextIteratorStreamer
from transformers.generation.stopping_criteria import EosTokenCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info
from threading import Thread


class MiMoVLInfer:
    def __init__(self, checkpoint_path, device='cuda', **kwargs):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype='auto', device_map=device, attn_implementation='flash_attention_2',
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)

    def __call__(self, inputs: dict, history: list = [], temperature: float = 1.0):
        messages = self.construct_messages(inputs)
        updated_history = history + messages
        text = self.processor.apply_chat_template(updated_history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(updated_history)
        model_inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt'
        ).to(self.model.device)
        tokenizer = self.processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            'max_new_tokens': 16000,
            'streamer': streamer,
            'stopping_criteria': StoppingCriteriaList([EosTokenCriteria(eos_token_id=self.model.config.eos_token_id)]),
            'pad_token_id': self.model.config.eos_token_id,
            **model_inputs
        }
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        partial_response = ""
        for new_text in streamer:
            partial_response += new_text
            yield partial_response, updated_history + [{
                'role': 'assistant',
                'content': [{
                    'type': 'text',
                    'text': partial_response
                }]
            }]

    def _is_video_file(self, filename):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
        return any(filename.lower().endswith(ext) for ext in video_extensions)

    def construct_messages(self, inputs: dict) -> list:
        content = []
        for i, path in enumerate(inputs.get('files', [])):
            if self._is_video_file(path):
                content.append({
                    "type": "video",
                    "video": f'file://{path}'
                })
            else:
                content.append({
                    "type": "image",
                    "image": f'file://{path}'
                })
        query = inputs.get('text', '')
        if query:
            content.append({
                "type": "text",
                "text": query,
            })
        messages = [{
            "role": "user",
            "content": content,
        }]
        return messages
