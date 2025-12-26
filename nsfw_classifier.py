from transformers import pipeline

NSFW_THRESHOLD = 0.85

class NSFW_Classifier:
    def __init__(self, model_name = 'Falconsai/nsfw_image_detection', device = 'cpu', thres = NSFW_THRESHOLD):
        self.classifier = pipeline("image-classification", model=model_name, device=device)
        self.thres = thres

    def score(self, img):
        nsfw_score = [x["score"] for x in self.classifier(img) if x["label"] == "nsfw"][0]
        return nsfw_score
    
    def is_valid(self, img):
        score_ = self.score(img)
        return score_ < self.thres

