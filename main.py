from pipeline import ImgGenPipeline



if __name__ == "__main__":
    img_model_name = '../../ckpts/SANA1.5_1.6B_1024px_diffusers'

    pipe = ImgGenPipeline(generator_name='sana', generator_model=img_model_name, enable_nsfw_filter=False)
    pipe.draw('三体中的二向箔摧毁一台电脑', save_path='1.png')