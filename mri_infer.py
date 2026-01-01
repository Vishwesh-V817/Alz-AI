import os
import torch
import zipfile
import shutil
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torchvision import transforms
from attention import DualAttentionResNet18MRI 

class AlzheimerInferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DualAttentionResNet18MRI(num_classes=3).to(self.device)
        
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _normalize_and_format(self, slice_array):
        p1, p99 = np.percentile(slice_array, [1, 99])
        slice_array = np.clip(slice_array, p1, p99)
        slice_array = (slice_array - p1) / (p99 - p1 + 1e-8) * 255.0
        return slice_array.astype(np.uint8)

    def predict(self, file_path):
        slices_tensors = []
        
        # Check if it's a ZIP or a Single Image
        if not zipfile.is_zipfile(file_path):
            # --- CASE 1: SINGLE IMAGE FILE ---
            print("Processing as Single Image...")
            img = Image.open(file_path).convert("L")
            slices_tensors = [self.transform(img)] * 16
        else:
            # --- CASE 2: ZIP FILE (DICOMs OR PNG/JPG) ---
            tmp = "temp_inf_extract"
            if os.path.exists(tmp): shutil.rmtree(tmp)
            os.makedirs(tmp)
            
            with zipfile.ZipFile(file_path, 'r') as z:
                z.extractall(tmp)
            
            # Get all valid files in the zip
            all_files = []
            for root, _, fs in os.walk(tmp):
                for f in fs:
                    if not f.startswith('.') and f.lower().endswith(('.dcm', '.png', '.jpg', '.jpeg')):
                        all_files.append(os.path.join(root, f))
            
            if not all_files:
                shutil.rmtree(tmp)
                raise ValueError("No valid MRI files (.dcm, .png, .jpg) found in ZIP.")

            # DETECT TYPE: Are there DICOMs?
            is_dcm = any(f.lower().endswith('.dcm') for f in all_files)

            if is_dcm:
                # --- CASE 2A: DICOM 3-STEP PIPELINE ---
                print("Processing as ZIP of DICOMs...")
                dcm_folder = os.path.dirname(next(f for f in all_files if f.endswith('.dcm')))
                reader = sitk.ImageSeriesReader()
                dcm_names = reader.GetGDCMSeriesFileNames(dcm_folder)
                reader.SetFileNames(dcm_names)
                image = reader.Execute()
                reoriented_img = sitk.DICOMOrient(image, 'RAI')
                array = sitk.GetArrayFromImage(reoriented_img)
                
                # Step 2: 60-slice window
                total_slices = array.shape[0]
                if total_slices >= 250: start, end = 100, 160
                elif 180 <= total_slices <= 200: start, end = 50, 110
                else:
                    mid = total_slices // 2
                    start, end = max(0, mid - 30), min(total_slices, mid + 30)
                
                sixty_slices = array[start:end]
                # Step 3: Reverse
                sixty_slices = sixty_slices[::-1]
                
                # Step 4: Final 16 from middle
                mid_60 = len(sixty_slices) // 2
                selected_16 = sixty_slices[mid_60-8 : mid_60+8]
                
                # Pad if necessary
                if len(selected_16) < 16:
                    selected_16 = np.pad(selected_16, ((0, 16-len(selected_16)), (0,0), (0,0)), mode='edge')

                for s in selected_16:
                    norm_s = self._normalize_and_format(s)
                    slices_tensors.append(self.transform(Image.fromarray(norm_s)))
            
            else:
                # --- CASE 2B: ZIP OF IMAGES (PNG/JPG) ---
                print("Processing as ZIP of Standard Images...")
                all_files.sort() # Ensure anatomical order
                mid = len(all_files) // 2
                # Take middle 16
                selected = all_files[mid-8 : mid+8] if len(all_files) >= 16 else all_files + [all_files[-1]]*(16-len(all_files))
                
                for f in selected:
                    img = Image.open(f).convert("L")
                    slices_tensors.append(self.transform(img))

            shutil.rmtree(tmp)

        # Final Processing
        input_tensor = torch.stack(slices_tensors).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        labels = ["0_Normal", "1_Mild_Impairment", "2_Moderate_Impairment"]
        return {
            "prediction": labels[pred_idx],
            "confidence": float(probs[0][pred_idx]),
            "risk_level": pred_idx,
            "all_probs": {labels[i]: float(probs[0][i]) for i in range(3)}
        }