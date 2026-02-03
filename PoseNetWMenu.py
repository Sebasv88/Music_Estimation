#!/usr/bin/env python3
import pygame
import sys
import time
import numpy as np
import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

# Inicializa pygame
pygame.init()

# Configuración ventana principal
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("PoseNet IMX500 - Selector Keypoints")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

# Nombres de los 17 keypoints
KEYPOINT_NAMES = [
    "Nariz", "Ojo_Izq", "Ojo_Der", "Oreja_Izq", "Oreja_Der",
    "Hombro_Izq", "Hombro_Der", "Codo_Izq", "Codo_Der",
    "Muñeca_Izq", "Muñeca_Der", "Cadera_Izq", "Cadera_Der",
    "Rodilla_Izq", "Rodilla_Der", "Tobillo_Izq", "Tobillo_Der"
]

class PoseNetApp:
    def __init__(self):
        self.model_path = "/usr/share/imx500-models/imx500_network_posenet.rpk"
        self.imx500 = IMX500(self.model_path)
        self.picam2 = Picamera2(self.imx500.camera_num)
        
        # Estado del menú
        self.menu_active = True
        self.selected_kp = -1  
        self.keypoints_data = [] 
        
        self.setup_camera()
    
    def setup_camera(self):
        config = self.picam2.create_preview_configuration(main={"size": (WINDOW_WIDTH, WINDOW_HEIGHT)})
        self.picam2.configure(config)
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(config, show_preview=False)
        self.imx500.set_auto_aspect_ratio()
        
        # Callback para procesar keypoints
        self.picam2.pre_callback = self.pose_callback
    
    def pose_callback(self, request: CompletedRequest):
        """Captura keypoints"""
        np_outputs = self.imx500.get_outputs(request.get_metadata(), add_batch=True)
        if np_outputs and len(np_outputs) > 0:
            heatmaps = np_outputs[0][0]  # (23,31,17)
            self.parse_keypoints(heatmaps)
    
    def parse_keypoints(self, heatmaps):
        """Parsea heatmaps → lista keypoints"""
        h_map, w_map, n_kp = heatmaps.shape
        self.keypoints_data = []
        
        for kp_id in range(n_kp):
            heatmap = heatmaps[:, :, kp_id]
            y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            conf_raw = float(heatmap[y_max, x_max])
            conf = max(0.0, conf_raw)
            
            if conf > 0.15:
                x_norm = x_max / w_map
                y_norm = y_max / h_map
                self.keypoints_data.append({
                    'id': kp_id,
                    'name': KEYPOINT_NAMES[kp_id],
                    'x': x_norm,
                    'y': y_norm,
                    'conf': conf
                })
    
    def draw_pose_overlay(self, surface):
        """Dibuja keypoints"""
        h, w = WINDOW_HEIGHT, WINDOW_WIDTH
        
        for kp_data in self.keypoints_data:
            if self.selected_kp == -1 or kp_data['id'] == self.selected_kp:
                x_px = int(kp_data['x'] * w)
                y_px = int(kp_data['y'] * h)
                
                # Color especial para seleccionado
                color = BLUE if kp_data['id'] == self.selected_kp else GREEN
                size = 8 if kp_data['id'] == self.selected_kp else 10
                
                pygame.draw.circle(surface, color, (x_px, y_px), size)
                font = pygame.font.Font(None, 24)
                text = font.render(f"{kp_data['id']}", True, WHITE)
                surface.blit(text, (x_px + 15, y_px - 10))
    
    def draw_menu(self, surface):
        """Dibuja menú pygame"""
        # Fondo semitransparente
        overlay = pygame.Surface((300, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))
        
        # Título
        font_title = pygame.font.Font(None, 48)
        title = font_title.render("SELECCIONA KP", True, WHITE)
        surface.blit(title, (20, 20))
        
        # Botones keypoints (grid 4x5)
        button_rects = []
        for i in range(17):
            row, col = divmod(i, 4)
            x = 20 + col * 70
            y = 100 + row * 50
            
            rect = pygame.Rect(x, y, 65, 45)
            
            if i == self.selected_kp:
                color = BLUE
            else: 
                color = GRAY
                
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, WHITE, rect, 2)
            
            # Texto keypoint
            font = pygame.font.Font(None, 28)
            text1 = font.render(str(i), True, WHITE)
            text2 = pygame.font.Font(None, 20).render(KEYPOINT_NAMES[i][:3], True, WHITE)
            
            surface.blit(text1, (x+25, y+10))
            surface.blit(text2, (x+15, y+28))
            
            button_rects.append((rect, i))
        
        # Botón "Todos"
        todos_rect = pygame.Rect(20, 550, 260, 50)
        pygame.draw.rect(surface, GREEN if self.selected_kp is None else GRAY, todos_rect)
        pygame.draw.rect(surface, WHITE, todos_rect, 2)
        todos_text = pygame.font.Font(None, 36).render("TODOS", True, WHITE)
        surface.blit(todos_text, (80, 565))
        button_rects.append((todos_rect, -1))
        
        # Info inferior
        info_font = pygame.font.Font(None, 24)
        info = info_font.render(f"Keypoints detectados: {len(self.keypoints_data)}", True, WHITE)
        surface.blit(info, (20, 620))
        
        return button_rects
    
    def run(self):
        """Bucle principal"""
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                if event.type == pygame.MOUSEBUTTONDOWN and self.menu_active:
                    mouse_pos = pygame.mouse.get_pos()
                    button_rects = self.draw_menu(screen)
                    
                    for rect, kp_id in button_rects:
                        # Comprueba si el usuario esta dentro del rectangulo
                        if rect.collidepoint(mouse_pos):
                            self.selected_kp = kp_id
                            print(f"Keypoint seleccionado: {kp_id} ({KEYPOINT_NAMES[kp_id] if kp_id >= 0 else 'TODOS'})")
            
            # Capturar frame cámara
            request = self.picam2.capture_request()
            frame = request.make_array("main")
            request.release()
            
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            frame_rgb = np.ascontiguousarray(frame_rgb)
            frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
            frame_pygame = pygame.transform.scale(frame_pygame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            # Copia la imagen frame_pygame en la esquina superior izquierda (0,0)
            screen.blit(frame_pygame, (0, 0))
            
            # Dibuja keypoints
            self.draw_pose_overlay(screen)
            
            # Dibuja menu si activo
            if self.menu_active:
                button_rects = self.draw_menu(screen)
            
            # Muestra la ventana todo el contenido
            pygame.display.flip()
            clock.tick(30)
        
        self.picam2.stop()
        self.imx500.stop()
        pygame.quit()

if __name__ == "__main__":
    app = PoseNetApp()
    app.run()
