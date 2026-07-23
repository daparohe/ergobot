# Bot de Análisis de Postura Ergonómica

Bot de Telegram que analiza fotografías y clasifica la postura corporal del usuario en 6 categorías ergonómicas (3 correctas / 3 incorrectas), usando detección de pose con **MediaPipe** y un clasificador **MLP (Keras → TFLite)** entrenado sobre ángulos articulares y coordenadas normalizadas de keypoints. Corre en tiempo real sobre una **Raspberry Pi** como servicio `systemd`.

> Proyecto desarrollado como parte de una investigación sobre prevención de lesiones músculo-esqueléticas mediante inteligencia artificial aplicada a la ergonomía.

---

## Tabla de contenido

- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Clases detectadas](#clases-detectadas)
- [Dataset y extracción de características](#dataset-y-extracción-de-características)
- [Modelo](#modelo)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Despliegue como servicio (Raspberry Pi)](#despliegue-como-servicio-raspberry-pi)
- [Reentrenar el modelo](#reentrenar-el-modelo)
- [Licencia](#licencia)
- [Citación](#citación)

---

## Arquitectura del sistema

```
Foto (Telegram) → OpenCV (decode) → MediaPipe Pose Landmarker
                                            │
                                            ▼
                              Extracción de 33 keypoints
                                            │
                                            ▼
                     Ingeniería de características (13 ángulos
                     articulares + 20 keypoints normalizados +
                     visibilidad promedio = 74 features)
                                            │
                                            ▼
                          StandardScaler (scaler.pkl)
                                            │
                                            ▼
                    Red neuronal MLP (TFLite) → 6 clases
                                            │
                                            ▼
              Mensaje de Telegram: clase + confianza + ángulos
              medidos + recomendaciones ergonómicas personalizadas
```

## Clases detectadas

| Clase | Descripción |
|---|---|
| ✅ BIEN DE PIE | Columna vertical, cabeza alineada, peso distribuido |
| ✅ BIEN LEVANTANDO | Rodillas flexionadas, espalda recta, carga cerca del cuerpo |
| ✅ BIEN SENTADOS | Espalda apoyada, caderas ~90°, pies apoyados |
| ⚠️ MAL DE PIE | Columna inclinada, cabeza adelantada, hombros caídos |
| ⚠️ MAL LEVANTANDO | Espalda doblada, rodillas extendidas al levantar peso |
| ⚠️ MAL SENTADOS | Espalda sin apoyo, cuello inclinado, postura encorvada |

## Dataset y extracción de características

Las imágenes originales (organizadas en carpetas por clase) se procesaron con **MediaPipe Pose Landmarker (`pose_landmarker_full.task`)** para extraer 33 keypoints corporales por imagen. A partir de esos keypoints se calcularon, por cada fotografía:

- **13 ángulos articulares**: inclinación de columna, ángulo de cuello, ángulos de rodilla (izq/der), ángulos de codo (izq/der), inclinación de hombros, inclinación de caderas, ángulos de cadera (izq/der), adelanto de cabeza, altura de cadera relativa, asimetría de hombros.
- **20 coordenadas de keypoints** (x, y de 10 puntos clave), normalizadas respecto al centro de caderas y a la distancia hombro-cadera, para hacerlas invariantes a la posición y escala del sujeto en la imagen.
- **1 valor de visibilidad promedio** de los puntos críticos del tronco.

Total: **59 columnas** (incluye metadatos) → **1,893 filas** procesadas correctamente sobre el total de fotografías capturadas por clase (ver `train/dataset_info.txt` para el desglose exacto y la tasa de fotos sin pose detectada por clase).

El script de extracción está documentado en `train/modelo.ipynb`.

## Modelo

Clasificador **MLP secuencial** (Keras), entrenado sobre las 74 características derivadas (no directamente sobre las imágenes):

```
Input(74)
 → Dense(256) + BatchNorm + ReLU + Dropout(0.4)
 → Dense(128) + BatchNorm + ReLU
 → Dense(64)  + ReLU + Dropout(0.3)
 → Dense(6, softmax)
```

- Regularización L2 (0.001) en todas las capas densas.
- Optimizador Adam, `sparse_categorical_crossentropy`.
- `class_weight` balanceado (dataset con clases desbalanceadas, ver tabla de conteos).
- Early stopping sobre `val_accuracy` (paciencia 25 épocas, máx. 200 épocas).
- Split: 70% train / 15% val / 15% test, estratificado, `random_state=42`.
- Conversión final a **TFLite** (`Optimize.DEFAULT`) para inferencia en Raspberry Pi.

Métricas de test, matriz de confusión y curvas de entrenamiento completas: ver `train/reporte_entrenamiento.txt` y `train/curvas_entrenamiento.png`.

## Estructura del repositorio

```
postura-ergonomica-bot/
├── train/          # Notebook de extracción de features + entrenamiento
├── model/           # Artefactos entrenados (tflite, scaler, label encoder)
├── bot/              # Bot de Telegram + configuración de despliegue
└── docs/            # Diagramas y capturas de apoyo
```

## Instalación

```bash
git clone https://github.com/daparohe/ergobot.git
cd postura-ergonomica-bot
pip install -r requirements.txt
```

Descarga el modelo de detección de pose de MediaPipe (no incluido en el repo por tamaño y por ser un artefacto de terceros):

```bash
wget -O bot/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

> En Raspberry Pi con recursos limitados, MediaPipe también ofrece una variante `pose_landmarker_lite.task`, más liviana.

## Configuración

Crea un bot en Telegram con [@BotFather](https://t.me/BotFather) y copia el token. Luego:

```bash
TELEGRAM_TOKEN=tu_token_aqui
```

## Uso

```bash
cd bot
python bot_posturas.py
```

Comandos disponibles en el bot:
- `/start` — menú principal
- `/info` — qué hace el bot
- `/uso` — instrucciones de uso
- `/clases` — descripción de las 6 posturas detectadas

Basta con enviar una foto al chat para recibir el análisis.

## Despliegue como servicio (Raspberry Pi)

El archivo `bot/postura-bot.service` permite correr el bot como servicio persistente con `systemd`, con reinicio automático ante fallos:

```bash
sudo cp bot/postura-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable postura-bot
sudo systemctl start postura-bot
sudo journalctl -u postura-bot -f   # ver logs en vivo
```

Recuerda editar las rutas (`WorkingDirectory`, `ExecStart`) y el usuario dentro del archivo `.service` según tu instalación.

## Reentrenar el modelo

Todo el pipeline (extracción de keypoints → construcción del CSV → entrenamiento → exportación a TFLite) está en `train/modelo.ipynb`. Para reentrenar con tu propio dataset de fotografías:

1. Organiza las imágenes en carpetas por clase (una carpeta por cada una de las 6 posturas).
2. Ejecuta la celda de extracción de keypoints (genera `dataset_posturas.csv` y `dataset_info.txt`).
3. Ejecuta la celda de entrenamiento (genera el `.keras`, `.tflite`, `scaler.pkl`, `label_encoder.pkl` y el reporte de métricas).

## Licencia

<!-- Elige una licencia (MIT es habitual para este tipo de proyectos académicos abiertos) y agrégala como archivo LICENSE en la raíz del repo. -->
Este proyecto se distribuye bajo licencia [MIT](LICENSE). El modelo `pose_landmarker_full.task` es propiedad de Google y se distribuye bajo sus propios términos (ver [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)).

## Citación

Si este trabajo es útil en tu investigación, por favor cítalo:

```bibtex
@misc{posturaergonomica2026,
  title   = {Bot de Análisis de Postura Ergonómica basado en MediaPipe y Redes Neuronales},
  author  = {David Rosales, Leydi Mingo, Cristhian Prieto},
  year    = {2026},
  howpublished = {\url{https://github.com/daparohe/ergobot}}
}
```
