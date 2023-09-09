from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
)
import requests


# TrOCR is a decoder model and should be used within a VisionEncoderDecoderModel
# init vision2text model with random weights
encoder = ViTModel(ViTConfig())
decoder = TrOCRForCausalLM(TrOCRConfig())
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# If you want to start from the pretrained model, load the checkpoint with `VisionEncoderDecoderModel`
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
url = "https://raw.githubusercontent.com/dusklight00/ai-evaluator/main/versatile.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
text = "Concrete is a versatile composed of a mixture of cement aggregates (such as sand and gravel) and water. There are several different types of concrete, such designed to meet specific requirements based on intended use."

# training
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

labels = processor.tokenizer(text, return_tensors="pt").input_ids
outputs = model(pixel_values, labels=labels)
loss = outputs.loss
round(loss.item(), 2)

# inference
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
generated_text