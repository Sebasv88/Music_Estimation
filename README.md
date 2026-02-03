# MUSIC ESTIMATION
Sistema interactivo que utiliza estimación de poses para la reproducción de sonidos de instrumentos musicales moviendo articulaciones específicas, según la movilidad motora de la persona en cuestión.

Este proyecto combina visión por computadora y música generativa: detecta la pose corporal de una persona por defecto y, únicamente al seleccionar un determinado conjunto {keypoint, instrumento} desde el menú principal, genera música en tiempo real con el movimiento.

🚀 CARACTERÍSTICAS
- Detecta y sigue articulaciones corporales usando Pose Estimation.
- Proporciona un menú interactivo con diferentes opciones:
    - Selección de keypoint (articulación).
    - Selección de instrumento {Guitarra, Piano, Flauta}
    - Selección por defecto (pose corporal completa = 17 keypoints).
- Divide la pantalla en cuatro regiones según las melodías especificadas {Do, Re, Mi, Fa}.
- Genera los sonidos del instrumento en el movimiento, al cruzar de una zona a otra.
- Destaca la zona actual proporcionando un contorno diferente.

📦 ESTRUCTURA

<pre> 
    Music_Estimation/ 
    ├── PoseNet.py 
    ├── PoseNetWMenu.py 
    ├── PoseNetWMenuWMusic.py 
    ├── instrumentos/ 
    ├── piCamera/ 
    ├── Icon/ 
    └── README.md 
</pre>

- PoseNet.py: implementación base de estimación de poses, a partir del modelo PoseNet.
- PoseNetWMenu.py: versión con menú gráfico para la selección de un keypoint determinado.
- PoseNetWMenuWMusic.py: versión extendida que permite la selección de un instrumento y reproducción de cuatro melodías definidas en Music_Estimation/instrumentos {Do, Re, Mi, Fa} al mover la articulación seleccionada.

📥 INSTALACIÓN

Clona este repositorio:
    
    git clone https://github.com/Sebasv88/Music_Estimation.git
    cd Music_Estimation

Instala dependencias:

FW del IMX500

    sudo apt install imx500-all

OpenCV

    sudo apt install python3-opencv python3-munkres

Picamera2 (si no está pre-instalado, normalmente en las placas raspberry pi que lo soportan ya lo tienen instalado)

    sudo apt install python3-picamera2 --no-install-recommends

PyGame

    sudo apt install python3-pygame

Busca el modelo PoseNet imx500_network_posenet.rpk en la siguiente ruta /usr/share/imx500-models/

🧪 USO

Ejecuta uno de los scripts principales:

    python PoseNetWMenuWMusic.py

1. La aplicación muestra la venta de vista previa y un menú interactivo en la parte lateral izquierda.
2. Por defecto realiza la estimación de poses por completo, es decir, refleja todos los keypoints detectados.
3. Selecciona con el ratón un determinado keypoint según la movilidad motora del usuario. El determinado es identificado con un color diferente.
4. Si se ha especificado una articulación, un submenú pasa a mostrarse, permitiendo la selección de un instrumento.
5. Una vez escogido el preferido, la aplicación reproduce la nota musical en base a la posición espacial (x,y) de la articulación en cuestión y la zona contenida. 
6. El botón "TODOS" devulve la aplicación a su estado original, suprimiendo la parte musical y el menú en cuestión.

🔧 PERSONALIZACIÓN

Puedes ajustar:
- Instrumentos sonoros: sustituye los archivos de sonido en la carpeta instrumentos/.
- Zonas de activación: cambia o modifica la regiones para reproducir más o menos melodías.
- Modelo de pose: sustituye o mejora el modelo de estimación por otro más avanzado.


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
