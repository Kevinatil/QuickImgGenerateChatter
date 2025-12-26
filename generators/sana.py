import torch
from diffusers import SanaPipeline


class SanaGenerator:
    def __init__(self, model_name: str, device: str):
        self.device = device

        self.pipe = SanaPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

        self.generator = torch.Generator(device=self.device)

        self.num_steps = 20
        self.guidance = 4.5

    def _generatable(self, **kwargs):
        if int(kwargs['width']) < 256 or int(kwargs['height']) < 256:
            print('width and height must larger than 256')
            return False
        return True

    @torch.inference_mode()
    def generate(
        self,
        width,
        height,
        prompt,
        seed = None
    ):
        if not self._generatable(width = width, height = height):
            print('not generatable')
            raise RuntimeError
        
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                seed = torch.Generator(device="cpu").seed()
        else:
            seed = torch.Generator(device="cpu").seed()


        img = self.pipe(prompt=prompt, height=height, width=width, guidance_scale=self.guidance, 
                        num_inference_steps=self.num_steps, generator=self.generator.manual_seed(seed))[0][0]

        return img


