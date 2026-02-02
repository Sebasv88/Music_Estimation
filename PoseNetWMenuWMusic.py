#!/usr/bin/env python3
import pygame
import sys
import time
import numpy as np
import os
import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500

# Inicializar pygame + mixer
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PREV_WINDOW_WIDTH, PREV_WINDOW_HEIGHT = 480, 270
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Music Estimation  - Keypoint + Instrumentos")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# Keypoints COCO
KEYPOINT_NAMES = ["Nariz","Ojo_Izq","Ojo_Der","Oreja_Izq","Oreja_Der",
                 "Hombro_Izq","Hombro_Der","Codo_Izq","Codo_Der",
                 "Muñeca_Izq","Muñeca_Der","Cadera_Izq","Cadera_Der",
                 "Rodilla_Izq","Rodilla_Der","Tobillo_Izq","Tobillo_Der"]

# Instrumentos disponibles (descarga sonidos WAV gratuitos)
INSTRUMENTS = [
    "Piano",
    "Guitarra", 
    "Violín",
    "Flauta"
]

# 4 melodías por zona (Do, Re, Mi, Fa)
MELODIES = {
    "Piano": ["C4", "D4", "E4", "F4"],
    "Guitarra": ["C3", "D3", "E3", "F3"],
    "Violín": ["C5", "D5", "E5", "F5"],
    "Flauta": ["C2", "D2", "E2", "F2"]
}

class PianoPoseApp:
    def __init__(self):
        self.model_path = "/usr/share/imx500-models/imx500_network_posenet.rpk"
        self.imx500 = IMX500(self.model_path)
        self.picam2 = Picamera2(self.imx500.camera_num)
        
        # Estado
        self.selected_kp = -1
        self.selected_inst = None
        self.keypoints_data = []
        self.current_zone = None
        self.prev_zone = None
        
        # Sonidos precargados
        self.sounds = {}
        self.load_sounds()
        
        self.setup_camera()
    
    def load_sounds(self):
        """Carga sonidos de instrumentos"""
        sound_dir = "/home/pi/Desktop/MusicEstimation/instrumentos"
        if not os.path.exists(sound_dir):
            os.makedirs(sound_dir)
            print("📁 Crea carpeta 'instrumentos/' y mete archivos WAV")
            return
        
        for instr, files in MELODIES.items():
            self.sounds[instr] = {}
            for note in files:
                sound_file = os.path.join(sound_dir, instr, f"{note}.wav")
                if os.path.exists(sound_file):
                    self.sounds[instr][note] = pygame.mixer.Sound(sound_file)
                    self.sounds[instr][note].set_volume(0.25)
    
    def setup_camera(self):
        # PyGame espera formato RGB => BGR88 proporciona salida ordenada en (RGB)
        config = self.picam2.create_preview_configuration(main={"size": (PREV_WINDOW_WIDTH, PREV_WINDOW_HEIGHT), "format": "BGR888"}, controls={'FrameRate': 20}, buffer_count=12)
        self.picam2.configure(config)
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(config, show_preview=False)
        self.imx500.set_auto_aspect_ratio()
        self.picam2.pre_callback = self.pose_callback
    
    def pose_callback(self, request):
        np_outputs = self.imx500.get_outputs(request.get_metadata(), add_batch=True)
        if np_outputs:
            heatmaps = np_outputs[0][0]
            self.parse_keypoints(heatmaps)
    
    def parse_keypoints(self, heatmaps):
        h_map, w_map, n_kp = heatmaps.shape
        self.keypoints_data = []
        
        for kp_id in range(n_kp):
            heatmap = heatmaps[:, :, kp_id]
            y_peak, x_peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            conf_raw = float(heatmap[y_peak, x_peak])
            conf = max(0.0, conf_raw)
            
            if conf > 0.15:
                x_norm = x_peak / w_map
                y_norm = y_peak / h_map
                self.keypoints_data.append({
                    'id': kp_id,
                    'name': KEYPOINT_NAMES[kp_id],
                    'x': x_norm, 
                    'y': y_norm, 
                    'conf': conf
                })
    
    def get_zone(self, x_norm):
        """Calcula zona vertical 0-3 según X del keypoint"""
        zona = None
        
        if x_norm is None:
            zona = x_norm
        else:
            # 0.0->0, 0.25->1, 0.5->2, 0.75->3
            zona = min(3, int(x_norm * 4))
            
        return zona
    
    def play_note(self, zone):
        """Toca nota según instrumento + zona"""
        if self.selected_inst and self.selected_kp >= 0:
            note = MELODIES[self.selected_inst][zone]
            if note in self.sounds[self.selected_inst]:
                self.sounds[self.selected_inst][note].play()
    
    def draw_zonas(self, surface):
        """Dibuja 4 zonas verticales"""
        zone_width = WINDOW_WIDTH // 4
        for i in range(4):
            x = i * zone_width
            color = YELLOW if i == self.current_zone else GRAY
            pygame.draw.rect(surface, color, (x, 0, zone_width, WINDOW_HEIGHT), 3)
            font = pygame.font.Font(None, 36)
            text = font.render(f"Zona {i}", True, WHITE)
            surface.blit(text, (x + zone_width//2 - 40, 20))
    
    def draw_pose_overlay(self, surface):
        h, w = WINDOW_HEIGHT, WINDOW_WIDTH
        
        for kp_data in self.keypoints_data:
            if self.selected_kp == -1 or kp_data['id'] == self.selected_kp:
                x_px = int(kp_data['x'] * w)
                y_px = int(kp_data['y'] * h)
                
                color = BLUE if kp_data['id'] == self.selected_kp else GREEN
                size = 10 if kp_data['id'] == self.selected_kp else 8
                
                pygame.draw.circle(surface, color, (x_px, y_px), size)
                font = pygame.font.Font(None, 28)
                text = font.render(f"{kp_data['id']}", True, WHITE)
                surface.blit(text, (x_px + 20, y_px - 15))
    
    def draw_menu_kp(self, surface):
        """Dibuja menú pygame"""
        # Fondo semitransparente
        overlay = pygame.Surface((320, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))
        
        # Título
        font_title = pygame.font.Font(None, 50)
        title = font_title.render("POSE MUSIC", True, WHITE)
        surface.blit(title, (20, 20))
        
        # Botones keypoints (grid 4x5)
        button_rects = []
        for i in range(17):
            row, col = divmod(i, 4)
            x = 20 + col * 70
            y = 70 + row * 50
            
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
        
        todos_rect = pygame.Rect(20, 650, 280, 50)
        pygame.draw.rect(surface, GREEN if self.selected_kp == -1 else GRAY, todos_rect)
        pygame.draw.rect(surface, WHITE, todos_rect, 2)
        todos_text = pygame.font.Font(None, 36).render("TODOS", True, WHITE)
        surface.blit(todos_text, (120, 675))
        button_rects.append((todos_rect, -1))

        # Info estado
        font_info = pygame.font.Font(None, 28)

        info3 = font_info.render(f"KP detectados: {len(self.keypoints_data)}", True, WHITE)
        surface.blit(info3, (20, 340))

        return button_rects
        
    def draw_menu_inst(self, surface):
        font_instr = pygame.font.Font(None, 32)
        title_instr = font_instr.render("Instrumento:", True, YELLOW)
        surface.blit(title_instr, (20, 370))
        
        button_inst = []    
        for id_inst, instr in enumerate(INSTRUMENTS):
            y = 410 + id_inst * (40 + 10)
            rect = pygame.Rect(20, y, 260, 40)
            color = GREEN if instr == self.selected_inst else GRAY
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, WHITE, rect, 2)
                
            text = font_instr.render(instr, True, WHITE)
            surface.blit(text, (40, y+8))
            
            button_inst.append((rect, instr))
        
        font_info = pygame.font.Font(None, 28)
        info2 = font_info.render(f"Instrumento: {self.selected_inst}", True, WHITE)
        surface.blit(info2, (20, 615))
            
        return button_inst
        
        '''
        if self.selected_kp >= 0:
            kp_name = KEYPOINT_NAMES[self.selected_kp]
            info1 = font_info.render(f"KP: {kp_name} ({self.selected_kp})", True, WHITE)
            surface.blit(info1, (20, 450))
        
        if self.selected_instrument:
            info2 = font_info.render(f"♪ {self.selected_instrument}", True, GREEN)
            surface.blit(info2, (20, 480))
        '''
        '''
        if self.current_zone is not None:
            info4 = font_info.render(f"Zona: {self.current_zone}", True, YELLOW)
            surface.blit(info4, (20, 550))
        '''
    
    def run(self):
        clock = pygame.time.Clock()
        
        button_rects_kp = []
        button_rects_inst = []
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    #button_rects = self.draw_menu_kp(screen)  # Recalcular rects
                    
                    for rect_kp, kp_id in button_rects_kp:
                        # Comprueba si el usuario esta dentro del rectangulo
                        if rect_kp.collidepoint(mouse_pos):
                            self.selected_kp = kp_id
                            print(f"Keypoint seleccionado: {kp_id} ({KEYPOINT_NAMES[kp_id] if kp_id >= 0 else 'TODOS'})")
                            
                            if kp_id == -1:
                                self.selected_inst = None
                                print(f"Ningun instrumento disponible")

                                
                    if self.selected_kp >= 0:
                        print(f"Se ha seleccionado un kp, dibujar menu inst")
                        for rect_ins, inst_ky in button_rects_inst:
                            # Comprueba si el usuario esta dentro del rectangulo
                            if rect_ins.collidepoint(mouse_pos):
                                self.selected_inst = inst_ky
                                print(f"Instrumento seleccionado: {inst_ky}")
        
            # Capturar frame
            request = self.picam2.capture_request()
            frame = request.make_array("main")
            request.release()
            
            # Conversión correcta => CV2 trabaja con BGR
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) => eliminamos una conversion de color (mejorando la fluidez)
            #frame_rgb = np.ascontiguousarray(frame_rgb)
            frame_rgb = np.ascontiguousarray(frame)
            frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
            frame_pygame = pygame.transform.scale(frame_pygame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            
            screen.blit(frame_pygame, (0, 0))
            
            # Detectar zona del keypoint seleccionado
            self.current_zone = None
            if self.selected_kp >= 0 and self.keypoints_data:
                for kp in self.keypoints_data:
                    if kp['id'] == self.selected_kp:
                        self.current_zone = self.get_zone(kp['x'])
                        break
            
            # Tocar sonido al cambiar zona
            if self.current_zone != self.prev_zone and self.current_zone is not None:
                self.play_note(self.current_zone)
            
            self.prev_zone = self.current_zone
            
            # Dibujar
            self.draw_pose_overlay(screen)
            self.draw_zonas(screen)
            button_rects_kp = self.draw_menu_kp(screen)
            
            if self.selected_kp >= 0:
                button_rects_inst = self.draw_menu_inst(screen)
                
            pygame.display.flip()
            clock.tick(20)
        
        self.picam2.stop()
        self.imx500.stop()
        pygame.quit()

# MEJORADOS MENUS, Y SONIDOS OK => MIRAR SI DESAPARECEN LOS SONIDOS AL PULSAR EL BOTON TODOS
# MEJORAR LO DE QUE NO ES FLUIDA LA PANTALLA
if __name__ == "__main__":
    app = PianoPoseApp()
    app.run()
