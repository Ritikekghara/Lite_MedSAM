# import os
# import io
# import base64
# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify, render_template
# from litemedsam.medsam_lite import MedSAM_Lite  # adjust this import path if needed
# import torch
# import torchvision.transforms as T
# import cv2

# app = Flask(__name__)

# # Load MedSAM model
# model = MedSAM_Lite("lite_medsam.pth")  # Make sure this checkpoint is correct
# model.eval()

# # Helper: Resize and pad image to 256x256
# def preprocess_image_and_box(image, bbox):
#     orig_w, orig_h = image.size
#     scale = 256 / max(orig_w, orig_h)
#     new_w, new_h = int(orig_w * scale), int(orig_h * scale)
#     resized = image.resize((new_w, new_h))
    
#     # Pad to 256x256
#     padded = Image.new("RGB", (256, 256))
#     padded.paste(resized, ((256 - new_w) // 2, (256 - new_h) // 2))

#     # Adjust bbox
#     x0, y0, x1, y1 = bbox
#     x0 = int(x0 * scale + (256 - new_w) // 2)
#     x1 = int(x1 * scale + (256 - new_w) // 2)
#     y0 = int(y0 * scale + (256 - new_h) // 2)
#     y1 = int(y1 * scale + (256 - new_h) // 2)

#     box_resized = [x0, y0, x1, y1]

#     return padded, box_resized, (orig_w, orig_h)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         image_file = request.files['image']
#         box_data = request.form['box']  # Expected as JSON-like string: [x0, y0, x1, y1]

#         image = Image.open(image_file).convert("RGB")
#         bbox = list(map(int, box_data.strip('[]').split(',')))

#         # Preprocess
#         processed_img, resized_box, original_size = preprocess_image_and_box(image, bbox)

#         img_tensor = T.ToTensor()(processed_img).unsqueeze(0)  # [1, 3, 256, 256]
#         resized_box = torch.tensor(resized_box).unsqueeze(0).float()  # [1, 4]

#         with torch.no_grad():
#             pred_mask = model(img_tensor, resized_box)[0][0].cpu().numpy()

#         # Resize mask back to original image size
#         mask_resized = cv2.resize((pred_mask > 0).astype(np.uint8) * 255, original_size[::-1])

#         # Encode mask to base64
#         mask_pil = Image.fromarray(mask_resized)
#         buffered = io.BytesIO()
#         mask_pil.save(buffered, format="PNG")
#         mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#         return jsonify({'mask': mask_base64})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# app.py
import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from litemedsam.medsam_lite import MedSAM_Lite
from litemedsam.utils import preprocess_image_and_box, overlay_mask_on_image
# from waitress import serve


app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedSAM_Lite("lite_medsam.pth").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        box = request.form.get('box')
        box = list(map(int, box.strip('[]').split(',')))
        image_bytes = file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        padded_img, resized_box, original_size = preprocess_image_and_box(image, box)

        image_tensor = torch.tensor(np.array(padded_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.
        image_tensor = image_tensor.to(device)

        box_tensor = torch.tensor([resized_box], dtype=torch.float).to(device)

        with torch.no_grad():
            image_embedding = model.image_encoder(image_tensor)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )

            mask = F.interpolate(low_res_masks, size=(256, 256), mode="bilinear", align_corners=False)
            mask = (mask > 0).float().cpu().numpy()[0, 0]

        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
        mask_resized = np.array(mask_img)
        overlay = overlay_mask_on_image(image, mask_resized)
        overlay_pil = Image.fromarray(overlay)

        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'overlay': encoded_image})

    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    # serve(app, host='0.0.0.0', port=5000)
