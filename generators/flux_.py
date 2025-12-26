import torch
from einops import rearrange
from PIL import Image

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool = True):
        assert model_name in ['flux-schnell', 'flux-dev']
        self.device = torch.device(device)
        self.offload = offload
        self.model_name = model_name
        self.is_schnell = (model_name == "flux-schnell")

        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            is_schnell=self.is_schnell,
        )

        self.num_steps = 4 if self.is_schnell else 50
        self.guidance = 3.5 # (1.0, 10.0, 3.5)

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

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=self.num_steps,
            guidance=self.guidance,
            seed=seed,
        )

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        # bring into PIL format
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img


if __name__ == "__main__":
    model_name = 'flux-dev'
    generator = FluxGenerator(model_name, 'cpu', True)
    is_schnell = model_name == "flux-schnell"

    prompt = 'computer in 23th century'

    width = 455
    height = 256
    seed = -1

    output = generator.generate(width, height, prompt, seed)
    output.save('img.png')