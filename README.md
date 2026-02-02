# MUSIC_ESTIMATION
Sistema interactivo que utiliza estimación de poses para reproducir sonidos de instrumentos musicales moviendo articulaciones específicas.

Este proyecto combina visión por computadora y música generativa: detecta la pose corporal de una persona (por ejemplo, manos, codos, etc.) y, al asociar movimientos de articulaciones concretas con sonidos de instrumentos, genera música en tiempo real.

🚀 CARACTERÍSTICAS
- Detecta y sigue articulaciones corporales usando Pose Estimation.
- Vincula movimientos de puntos clave con sonidos de instrumentos (piano, guitarra, batería, etc.).
- Modo con menú gráfico para seleccionar opciones y experimentar con distintos sonidos.
- Interfaz intuitiva para reproducir instrumentación con movimientos del cuerpo.

📦 ESTRUCTURA
Music_Estimation/
├── PoseNet.py  
├── PoseNetWMenu.py  
├── PoseNetWMenuWMusic.py  
├── Instrumentos/
    ├── V
    ├── G
    ├── F
    ├── A
├── PiCamera/
├── requeriments.txt  
└── README.md

- PoseNet.py: implementación base de estimación de poses.
- PoseNetWMenu.py: versión con menú gráfico para facilitar la interacción.
- PoseNetWMenuWMusic.py: versión extendida que asocia movimientos con reproducción de sonidos de instrumentos.

🧠 FUNCIONAMIENTO
1. Captura de vídeo: el sistema usa la cámara para capturar vídeo en tiempo real.
2. Estimación de pose: un modelo (por ejemplo, basado en PoseNet o MediaPipe) calcula las posiciones de articulaciones clave.
3. Mapeo a sonidos: al detectar que una articulación supera un umbral o cruza una zona definida, se reproduce un sonido de instrumento asignado.
4. Realimentación en tiempo real: puedes ver tu pose y experimentar con los sonidos al mover tus brazos o piernas.

📥 INSTALACIÓN
1. Clona este repositorio:
  git clone https://github.com/Sebasv88/Music_Estimation.git
  cd Music_Estimation

2. Crea un entorno virtual (opcional pero recomendado):
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows

3. Instala dependencias:
    pip install -r requirements.txt

4. Asegúrate de tener una cámara conectada si quieres usar funcionalidades en tiempo real.

🧪 USO
Ejecuta uno de los scripts principales:
    python PoseNetWMenuWMusic.py

- Selecciona en pantalla las opciones que desees (instrumentos, zonas de activación, sensibilidad, etc.).
- Muévete frente a la cámara y escucha cómo los sonidos responden a tus articulaciones.

🔧 PERSONALIZACIÓN

Puedes ajustar:
- Instrumentos sonoros: sustituye los archivos de sonido en la carpeta assets/sounds/.
- Zonas de activación: cambia las coordenadas o thresholds para hacer la experiencia más sensible o precisa.
- Modelo de pose: sustituye o mejora el modelo de estimación por otro más avanzado.

🧩 REQUISITOS
Este proyecto puede utilizar librerías como:
- opencv-python
- mediapipe / tensorflow / posenet (según implementación)
- pygame o similares para reproducir audio
(añade estas librerías en tu requirements.txt si no lo están)

Instálalas con:
    pip install opencv-python mediapipe pygame

📈 POSIBLES MEJORAS
- Añadir más instrumentos y sonidos personalizados.
- Integrar modelos más avanzados de pose (por ejemplo, detectores 3D).
- Añadir visualizaciones musicales sincronizadas con el movimiento.
- Aplicación móvil o web para que sea accesible desde cualquier dispositivo.

🤝 Contribuciones
¡Las contribuciones son bienvenidas!
Sigue estos pasos:
- Haz un fork del proyecto.
- Crea una nueva rama (git checkout -b feature/nueva-funcionalidad).
- Haz tus cambios y súbelos (git commit -m "Añadida nueva funcionalidad").
- Abre un pull request.
