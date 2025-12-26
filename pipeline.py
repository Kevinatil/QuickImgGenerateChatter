from prompter import Chatter
from nsfw_classifier import NSFW_Classifier

class ImgGenPipeline:
    def __init__(self, generator_name, generator_model, prompter_name='deepseek-r1:8b', enable_nsfw_filter = True, device='cpu'):
        assert generator_name in ['flux', 'sana']

        if generator_name == 'flux':
            assert generator_model in ['flux-dev', 'flux-schnell']
            from generators import FluxGenerator
            self.generator = FluxGenerator(model_name=generator_model, device=device, offload=True)
        elif generator_name == 'sana':
            from generators import SanaGenerator
            self.generator = SanaGenerator(model_name=generator_model, device=device)
        else:
            raise NotImplementedError
        
        self.prompter = Chatter(model_name=prompter_name)

        self.default_draw_prompt = '帮我写一个图像生成的专业高质量提示词，图像关于：{}。\n\n要求写实风格，发挥想象力，描述细致具体，图像的氛围需要和提示词一致。禁止输出：昆虫、蠕虫、排泄物、呕吐物等恶心的内容，禁止生成可能引发密集恐惧症的内容。提示词请只输出英文版本，禁止出现中文。只输出提示词即可，禁止输出其他内容。'
        self.draw_prompt = self.default_draw_prompt

        self.enable_nsfw_filter = enable_nsfw_filter
        if self.enable_nsfw_filter:
            self.nsfw_filter = NSFW_Classifier(device=device)

    def get_cur_draw_prompt(self):
        return self.draw_prompt
    
    def set_draw_prompt(self, prompt):
        self.draw_prompt = prompt

    def reset_draw_prompt(self):
        self.draw_prompt = self.default_draw_prompt

    def draw(self, query, width = 1024, height = 1024, seed = None, save_path = None):
        prompt, messages = self.prompter.chat_no_rag(self.draw_prompt.format(query))
        print('生成的提示词：\n{}'.format(prompt))
        img = self.generator.generate(width=width, height=height, prompt=prompt, seed=seed)

        if self.enable_nsfw_filter:
            is_valid = self.nsfw_filter.is_valid(img)
            if not is_valid:
                print('img not valid')
                return None

        if save_path is not None:
            img.save(save_path)

        return img
