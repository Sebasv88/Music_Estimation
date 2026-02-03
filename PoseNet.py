#!/usr/bin/env python3
import time, sys
import numpy as np
import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics


last_keypoints = None
last_scores = None
WINDOW_SIZE_H_W = (1080, 1920)

def ai_output_tensor_parse(metadata):
    """Parsea TU modelo (igual estructura oficial)"""
    global last_keypoints, last_scores
    # 3 tensores
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    #print(f"Salida: {np_outputs} \n")
    
    if np_outputs is not None and len(np_outputs) >= 2:
        # TU POSE NET: tensor0=keypoints, tensor1=scores
        h_kp = np_outputs[0][0]
        #print(f"Heatmp kp: {h_kp} \n")
        #print(f"Heatmap kp shape: {h_kp.shape} \n") #(23,31,17)
        h_score = np_outputs[1][0]
        #print(f"Heatmap score: {h_score} \n")
        #print(f"Heatmap shape: {h_score.shape} \n") #(23,31,17)
        
        h_hm, w_hm, n_kp = h_kp.shape
        #print(f"Dimension Heatmap kp {h_hm, w_hm, n_kp} \n")
        keypoints = []
        
        for kp_id in range(n_kp):
                heatmap = h_kp[:,:,kp_id]
                #print(f"Heatmap shape individual: {heatmap.shape, kp_id} \n")
                y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                #print(f"Heatmap ind coord: {y_max, x_max} \n")
                conf = float(heatmap[y_max][x_max])
                confV2 = np.max(heatmap)
                #print(f"Heatmap ind conf: {conf} \n")
                #print(f"Heatmap ind confV2: {confV2} \n")
                
                if conf > 0.2:
                        x_norm = x_max / w_hm
                        y_norm = y_max / h_hm
                        #print(f"Heatmap ind coord: {x_max, y_max} \n")
                        keypoints.append([x_norm, y_norm, conf])
         
        #print(f"Keypoints totales: {keypoints} \n")
        #print(f"Keypoints len: {len(keypoints)} \n")

        if len(keypoints) > 0:
                keypoints_np = np.array(keypoints, dtype=np.float32)
                avg_conf = float(np.mean([kp[2] for kp in keypoints]))
                detected_kp = len(keypoints)
                last_keypoints = keypoints_np.reshape(1, detected_kp, 3)
                last_scores = np.array([avg_conf])
        
        else:
                last_keypoints = None
                last_scores = None
    
    return last_scores, last_keypoints  # Igual que oficial (boxes=None)

# OK
def ai_output_tensor_draw(request: CompletedRequest, scores, keypoints, stream='main'):
    """Dibuja TU pose (adaptado del oficial)"""
    with MappedArray(request, stream) as m:
        if keypoints is not None and len(keypoints) > 0:
            h, w = m.array.shape[:2]
            #print(f"Array shape: {m.array.shape}; h,w: {h, w} \n")
            kp = keypoints[0]
            #print(f"Kp: {kp, kp.shape} \n")
            
            kp_list = kp.tolist()
            print(f"Kp_List: {kp_list} \n")
            
            for i, (x, y, conf) in enumerate(kp_list):
                # Igual que el .json
                if conf > 0.2:
                    # Escalado
                    x_px, y_px = int(x * w), int(y * h)
                    color = (0, 255, 0) if i < 5 else (255, 0, 0)
                    cv2.circle(m.array, (x_px, y_px), 8, color, -1)
                    cv2.putText(m.array, str(i), (x_px+10, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
            cv2.putText(m.array, f"Kp: {kp.shape[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def picamera2_pre_callback(request: CompletedRequest):
    """Callback idéntico al oficial"""
    scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    #print(f"Puntuacion: {scores} \n")
    #print(f"Keypoints: {keypoints} \n")
    #print(f"TENSOR PARSE OK, ENTRANDO EN DRAW...")
    ai_output_tensor_draw(request, scores, keypoints)

if __name__ == "__main__":
    model_path = "/usr/share/imx500-models/imx500_network_posenet.rpk"
    
    imx500 = IMX500(model_path)
    intrinsics = imx500.network_intrinsics
    
    if intrinsics.task != "pose estimation":
        print("No es modelo de pose estimation")
        sys.exit(1)
    
    # Configuración
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080)},
        controls={'FrameRate': 30}, buffer_count=12
    )
    
    # Inicio
    imx500.show_network_fw_progress_bar()
    # Mostrar ventana
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback
    
    while True:
        time.sleep(0.5)
   
