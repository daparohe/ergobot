"""
Bot de Telegram para análisis de postura ergonómica.
Corre en Raspberry Pi con modelo TFLite.

Requisitos:
    pip install python-telegram-bot mediapipe opencv-python numpy tensorflow

Archivos necesarios en la misma carpeta:
    pose_landmarker_full.task  (o _lite para RPi más antigua)
    modelo_posturas.tflite
    scaler.pkl
    label_encoder.pkl
"""

import io
import logging
import math
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
#import tensorflow as tf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)

# ─────────────────────────────────────────────
#  CONFIGURACIÓN — editar antes de desplegar
# ─────────────────────────────────────────────

TELEGRAM_TOKEN = "XXXXXXXXXX:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

MODEL_PATH    = Path("pose_landmarker_full.task")
TFLITE_PATH   = Path("modelo_posturas.tflite")
SCALER_PATH   = Path("scaler.pkl")
ENCODER_PATH  = Path("label_encoder.pkl")

MP_CONFIDENCE = 0.3

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CARGA DE MODELOS (una sola vez al arrancar)
# ─────────────────────────────────────────────

log.info("Cargando modelos...")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

try:
	import tflite_runtime.interpreter as tflite
except ImportError:
	import tensorflow.lite as tflite

interp = tflite.Interpreter(model_path=str(TFLITE_PATH))
interp.allocate_tensors()
inp_det = interp.get_input_details()
out_det = interp.get_output_details()

base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
mp_opts   = mp_vision.PoseLandmarkerOptions(
    base_options=base_opts,
    running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=MP_CONFIDENCE,
    min_pose_presence_confidence=MP_CONFIDENCE,
    min_tracking_confidence=MP_CONFIDENCE,
    num_poses=1,
)
detector = mp_vision.PoseLandmarker.create_from_options(mp_opts)

log.info("Modelos cargados correctamente.")

# ─────────────────────────────────────────────
#  LANDMARKS
# ─────────────────────────────────────────────

LM = {
    "NOSE":0,"LEFT_EYE":2,"RIGHT_EYE":5,"LEFT_EAR":7,"RIGHT_EAR":8,
    "LEFT_SHOULDER":11,"RIGHT_SHOULDER":12,"LEFT_ELBOW":13,"RIGHT_ELBOW":14,
    "LEFT_WRIST":15,"RIGHT_WRIST":16,"LEFT_HIP":23,"RIGHT_HIP":24,
    "LEFT_KNEE":25,"RIGHT_KNEE":26,"LEFT_ANKLE":27,"RIGHT_ANKLE":28,
    "LEFT_HEEL":29,"RIGHT_HEEL":30,"LEFT_FOOT_INDEX":31,"RIGHT_FOOT_INDEX":32,
}

# ─────────────────────────────────────────────
#  GEOMETRÍA
# ─────────────────────────────────────────────

def ang3(A, B, C):
    BA = np.array(A[:2]) - np.array(B[:2])
    BC = np.array(C[:2]) - np.array(B[:2])
    cos = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def angv(A, B):
    return math.degrees(math.atan2(abs(B[0]-A[0]), abs(B[1]-A[1]) + 1e-9))

def pm(A, B):
    return [(A[0]+B[0])/2, (A[1]+B[1])/2]

# ─────────────────────────────────────────────
#  EXTRACCIÓN DE FEATURES
# ─────────────────────────────────────────────

def extraer_features(pose_landmarks, w, h):
    lm = pose_landmarks
    def pt(i): p = lm[i]; return [p.x*w, p.y*h, p.z]
    def vis(i): return lm[i].visibility or 0.0

    CRITICOS = [LM["LEFT_SHOULDER"], LM["RIGHT_SHOULDER"], LM["LEFT_HIP"], LM["RIGHT_HIP"]]
    if any(vis(i) < 0.3 for i in CRITICOS):
        return None, None

    nariz = pt(LM["NOSE"])
    hiz   = pt(LM["LEFT_SHOULDER"]);  hde = pt(LM["RIGHT_SHOULDER"])
    ciz   = pt(LM["LEFT_HIP"]);       cde = pt(LM["RIGHT_HIP"])
    riz   = pt(LM["LEFT_KNEE"]);      rde = pt(LM["RIGHT_KNEE"])
    tiz   = pt(LM["LEFT_ANKLE"]);     tde = pt(LM["RIGHT_ANKLE"])
    eiz   = pt(LM["LEFT_ELBOW"]);     ede = pt(LM["RIGHT_ELBOW"])
    miz   = pt(LM["LEFT_WRIST"]);     mde = pt(LM["RIGHT_WRIST"])

    ch  = pm(hiz, hde)
    cc  = pm(ciz, cde)
    ref = np.linalg.norm(np.array(ch[:2]) - np.array(cc[:2])) + 1e-9

    # Valor por defecto 170° para articulaciones no visibles (típico de pie)
    def rodilla_val(cad, rod, tob, v_rod, v_tob):
        return ang3(cad, rod, tob) if vis(v_rod) > 0.3 and vis(v_tob) > 0.3 else 170.0

    def cadera_val(hom, cad, rod, v_cad, v_rod):
        return ang3(hom, cad, rod) if vis(v_cad) > 0.3 and vis(v_rod) > 0.3 else 170.0

    def codo_val(hom, cod, mun, v_cod, v_mun):
        return ang3(hom, cod, mun) if vis(v_cod) > 0.3 and vis(v_mun) > 0.3 else 170.0

    angulos = {
        "inclinacion_columna":    angv(ch, cc),
        "angulo_cuello":          ang3(nariz, ch, cc),
        "ang_rodilla_izq":        rodilla_val(ciz, riz, tiz, LM["LEFT_KNEE"],  LM["LEFT_ANKLE"]),
        "ang_rodilla_der":        rodilla_val(cde, rde, tde, LM["RIGHT_KNEE"], LM["RIGHT_ANKLE"]),
        "ang_codo_izq":           codo_val(hiz, eiz, miz, LM["LEFT_ELBOW"],  LM["LEFT_WRIST"]),
        "ang_codo_der":           codo_val(hde, ede, mde, LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"]),
        "inclinacion_hombros":    angv(hiz, hde),
        "inclinacion_caderas":    angv(ciz, cde),
        "ang_cadera_izq":         cadera_val(hiz, ciz, riz, LM["LEFT_HIP"],  LM["LEFT_KNEE"]),
        "ang_cadera_der":         cadera_val(hde, cde, rde, LM["RIGHT_HIP"], LM["RIGHT_KNEE"]),
        "adelanto_cabeza":        (nariz[0] - ch[0]) / ref,
        "altura_cadera_relativa": (cc[1] - tiz[1]) / (h + 1e-9) if vis(LM["LEFT_ANKLE"]) > 0.3
                                  else (cc[1] - tde[1]) / (h + 1e-9),
        "asimetria_hombros":      abs(hiz[1] - hde[1]) / ref,
    }

    cx, cy = cc[0], cc[1]
    kp = {}
    for nombre, idx in LM.items():
        p = lm[idx]
        kp[f"kp_{nombre.lower()}_x"] = (p.x*w - cx) / ref
        kp[f"kp_{nombre.lower()}_y"] = (p.y*h - cy) / ref

    vis_p = float(np.mean([vis(i) for i in [0,11,12,23,24,25,26]]))
    features = np.array(list(angulos.values()) + list(kp.values()) + [vis_p], dtype=np.float32)
    return features, angulos

# ─────────────────────────────────────────────
#  DIBUJAR KEYPOINTS SOBRE LA IMAGEN
# ─────────────────────────────────────────────

CONEXIONES = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (0,11),(0,12),
]

def dibujar_pose(img_rgb, pose_landmarks, w, h):
    img = img_rgb.copy()
    lm  = pose_landmarks
    for a, b in CONEXIONES:
        x1, y1 = int(lm[a].x*w), int(lm[a].y*h)
        x2, y2 = int(lm[b].x*w), int(lm[b].y*h)
        cv2.line(img, (x1,y1), (x2,y2), (100,200,255), 2)
    for i in range(33):
        px, py = int(lm[i].x*w), int(lm[i].y*h)
        cv2.circle(img, (px,py), 5, (255,220,0), -1)
        cv2.circle(img, (px,py), 5, (0,0,0), 1)
    return img

# ─────────────────────────────────────────────
#  GENERADOR DE CONSEJOS ESPECÍFICOS POR ÁNGULOS
# ─────────────────────────────────────────────

def generar_consejos(clase_nom: str, ang: dict) -> str:
    """
    Genera consejos personalizados según la clase detectada
    y los valores reales de los ángulos medidos.
    """
    col   = ang["inclinacion_columna"]
    cuell = ang["angulo_cuello"]
    hom   = ang["inclinacion_hombros"]
    adel  = ang["adelanto_cabeza"]
    asim  = ang["asimetria_hombros"]
    cad_i = ang["ang_cadera_izq"]
    cad_d = ang["ang_cadera_der"]
    rod_i = ang["ang_rodilla_izq"]
    rod_d = ang["ang_rodilla_der"]

    cad_p = np.mean([v for v in [cad_i, cad_d] if v > 0]) if any(v > 0 for v in [cad_i, cad_d]) else 170.0
    rod_p = np.mean([v for v in [rod_i, rod_d] if v > 0]) if any(v > 0 for v in [rod_i, rod_d]) else 170.0

    lineas = []

    if clase_nom == "BIEN DE PIE":
        lineas.append("✅ *Postura de pie correcta.*")
        if col < 3:
            lineas.append("• Columna perfectamente vertical ({:.1f}°). ¡Excelente!".format(col))
        if abs(adel) < 0.03:
            lineas.append("• Cabeza bien alineada sobre los hombros.")
        lineas.append("• Continúa manteniendo esta postura y descansa cada 30 min.")

    elif clase_nom == "MAL DE PIE":
        lineas.append("⚠️ *Postura de pie incorrecta. Correcciones:*")
        if col > 10:
            lineas.append("• ⬆️ Columna inclinada {:.1f}° — endereza la espalda, imagina un hilo tirando de tu cabeza hacia arriba.".format(col))
        elif col > 5:
            lineas.append("• ↕️ Ligera inclinación de columna ({:.1f}°) — intenta verticalizarla un poco más.".format(col))
        if adel > 0.1:
            lineas.append("• 👤 Cabeza adelantada ({:.2f}) — lleva la barbilla ligeramente hacia atrás, orejas sobre hombros.".format(adel))
        if asim > 0.15:
            lineas.append("• ↔️ Hombros asimétricos ({:.2f}) — nivela los hombros, uno está más alto que el otro.".format(asim))
        if hom > 20:
            lineas.append("• 🔄 Inclinación lateral de hombros ({:.1f}°) — verifica que no estés cargando peso de un solo lado.".format(hom))
        if cuell < 20:
            lineas.append("• 🔽 Cuello muy flexionado ({:.1f}°) — levanta la vista, no mires el suelo al estar de pie.".format(cuell))
        if cad_p < 155:
            lineas.append("• 🦵 Caderas flexionadas ({:.1f}°) — extiende las caderas, no te encojas hacia adelante.".format(cad_p))
        if not lineas[1:]:
            lineas.append("• Revisa la alineación general: cabeza, hombros, caderas y tobillos deben estar en línea vertical.")

    elif clase_nom == "BIEN SENTADOS":
        lineas.append("✅ *Postura sentada correcta.*")
        if col < 5:
            lineas.append("• Columna bien apoyada ({:.1f}° de inclinación). ¡Perfecto!".format(col))
        if 85 <= cad_p <= 100:
            lineas.append("• Ángulo de cadera ideal ({:.1f}°).".format(cad_p))
        lineas.append("• Recuerda levantarte y estirar cada 30 minutos.")

    elif clase_nom == "MAL SENTADOS":
        lineas.append("⚠️ *Postura sentada incorrecta. Correcciones:*")
        if col > 15:
            lineas.append("• ⬆️ Espalda muy inclinada ({:.1f}°) — apoya toda la espalda en el respaldo de la silla.".format(col))
        elif col > 8:
            lineas.append("• ↕️ Ligera inclinación de espalda ({:.1f}°) — acércate más al respaldo.".format(col))
        if adel > 0.1:
            lineas.append("• 👤 Cabeza adelantada ({:.2f}) — aleja la pantalla o súbela para no inclinar el cuello.".format(adel))
        if cuell < 15:
            lineas.append("• 🔽 Cuello muy inclinado ({:.1f}°) — la pantalla debe estar a la altura de tus ojos.".format(cuell))
        if cad_p < 75 or cad_p > 110:
            lineas.append("• 🦵 Ángulo de cadera fuera de rango ({:.1f}°, ideal 85-100°) — ajusta la altura de la silla.".format(cad_p))
        if asim > 0.15:
            lineas.append("• ↔️ Hombros asimétricos ({:.2f}) — siéntate centrado, no te recarges hacia un lado.".format(asim))
        if not lineas[1:]:
            lineas.append("• Apoya completamente la espalda, pies planos en el suelo y pantalla a la altura de los ojos.")

    elif clase_nom == "BIEN LEVANTANDO":
        lineas.append("✅ *Técnica de levantamiento correcta.*")
        if rod_p < 130:
            lineas.append("• Rodillas bien dobladas ({:.1f}°). ¡Así se hace!".format(rod_p))
        if col < 20:
            lineas.append("• Espalda relativamente recta al levantar ({:.1f}°).".format(col))
        lineas.append("• Mantén el objeto pegado al cuerpo durante todo el movimiento.")

    elif clase_nom == "MAL LEVANTANDO":
        lineas.append("⚠️ *Técnica de levantamiento incorrecta. Correcciones:*")
        if col > 30:
            lineas.append("• ⬆️ Espalda muy doblada ({:.1f}°) — el mayor error: dobla las RODILLAS, no la espalda.".format(col))
        elif col > 15:
            lineas.append("• ↕️ Espalda inclinada ({:.1f}°) — intenta mantenerla más recta al levantar.".format(col))
        if rod_p > 150:
            lineas.append("• 🦵 Rodillas casi rectas ({:.1f}°) — dóblalas más antes de levantar, usa la fuerza de las piernas.".format(rod_p))
        if adel > 0.15:
            lineas.append("• 👤 Cabeza muy adelantada ({:.2f}) — mira hacia adelante, no hacia abajo, al hacer el esfuerzo.".format(adel))
        if asim > 0.2:
            lineas.append("• ↔️ Asimetría alta ({:.2f}) — levanta el objeto de frente, no de costado.".format(asim))
        if not lineas[1:]:
            lineas.append("• Regla de oro: dobla rodillas, espalda recta, objeto cerca del cuerpo, exhala al levantar.")

    return "\n".join(lineas)

# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL DE ANÁLISIS
# ─────────────────────────────────────────────

def analizar_imagen(img_bytes: bytes) -> tuple[bytes | None, str]:
    """
    Recibe bytes de imagen, devuelve (imagen_con_keypoints_bytes, texto_respuesta).
    Si falla, devuelve (None, mensaje_de_error).
    """
    arr    = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "❌ No pude leer la imagen. Intenta con otro formato (JPG, PNG)."

    h, w   = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    resultado = detector.detect(mp_image)

    if not resultado.pose_landmarks:
        return None, "❌ No detecté ninguna persona en la imagen.\nAsegúrate de que la foto muestre el cuerpo completo de frente o de perfil."

    features, angulos = extraer_features(resultado.pose_landmarks[0], w, h)
    if features is None:
        return None, "❌ No pude ver claramente hombros y caderas.\nPrueba con una foto más abierta donde se vea el torso completo."

    # Inferencia
    X_sc = scaler.transform(features.reshape(1, -1)).astype(np.float32)
    interp.set_tensor(inp_det[0]["index"], X_sc)
    interp.invoke()
    probs     = interp.get_tensor(out_det[0]["index"])[0]
    clase_id  = int(np.argmax(probs))
    clase_nom = le.classes_[clase_id]
    confianza = float(probs[clase_id]) * 100

    # Imagen con keypoints
    img_vis  = dibujar_pose(img_rgb, resultado.pose_landmarks[0], w, h)
    img_out  = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    ok, buf  = cv2.imencode(".jpg", img_out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_bytes_out = bytes(buf) if ok else None

    # ── Construir mensaje ─────────────────────────────────────────────────
    es_buena = clase_nom.startswith("BIEN")
    emoji_res = "✅" if es_buena else "⚠️"

    # Resultado principal
    msg = f"{emoji_res} *{clase_nom}*  —  {confianza:.1f}% de confianza\n\n"

    # Probabilidades > 40%
    probs_filtradas = [(le.classes_[i], float(probs[i])*100) for i in range(len(le.classes_)) if probs[i] >= 0.40]
    probs_filtradas.sort(key=lambda x: x[1], reverse=True)
    if len(probs_filtradas) > 1:
        msg += "📊 *Probabilidades relevantes:*\n"
        for cn, pr in probs_filtradas:
            barra = "█" * int(pr / 10)
            msg  += f"  {cn}: {pr:.1f}% {barra}\n"
        msg += "\n"

    # Ángulos medidos
    col   = angulos["inclinacion_columna"]
    cuell = angulos["angulo_cuello"]
    hom   = angulos["inclinacion_hombros"]
    adel  = angulos["adelanto_cabeza"]
    asim  = angulos["asimetria_hombros"]
    cad_i = angulos["ang_cadera_izq"]
    cad_d = angulos["ang_cadera_der"]
    rod_i = angulos["ang_rodilla_izq"]
    rod_d = angulos["ang_rodilla_der"]

    def fmt_ang(v, umbral_ok, mayor_es_mejor):
        if v == 170.0:
            return "~170° (estimado)"
        ok = (v >= umbral_ok) if mayor_es_mejor else (abs(v) <= umbral_ok)
        icono = "🟢" if ok else "🔴"
        return f"{v:.1f}° {icono}"

    cad_p = np.mean([v for v in [cad_i, cad_d] if v > 0]) if any(v > 0 for v in [cad_i, cad_d]) else 170.0
    rod_p = np.mean([v for v in [rod_i, rod_d] if v > 0]) if any(v > 0 for v in [rod_i, rod_d]) else 170.0

    msg += "📐 *Ángulos medidos:*\n"
    msg += f"  Columna:      {fmt_ang(col,   5,   False)}\n"
    msg += f"  Cuello:       {fmt_ang(cuell, 30,  True)}\n"
    msg += f"  Cadera prom:  {fmt_ang(cad_p, 160, True)}\n"
    msg += f"  Rodilla prom: {fmt_ang(rod_p, 160, True)}\n"
    msg += f"  Hombros:      {fmt_ang(hom,   15,  False)}\n"
    msg += f"  Adelanto cab: {adel:.2f} {'🟢' if abs(adel)<=0.05 else '🔴'}\n"
    msg += f"  Asimetría:    {asim:.2f} {'🟢' if asim<=0.1 else '🔴'}\n\n"

    # Consejos personalizados
    msg += "💡 *Análisis y recomendaciones:*\n"
    msg += generar_consejos(clase_nom, angulos)

    return img_bytes_out, msg

# ─────────────────────────────────────────────
#  HANDLERS DE TELEGRAM
# ─────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    teclado = [
        [InlineKeyboardButton("ℹ️ ¿Para qué sirve este bot?", callback_data="info")],
        [InlineKeyboardButton("📖 ¿Cómo usarlo?",             callback_data="uso")],
        [InlineKeyboardButton("🎯 Clases que detecta",        callback_data="clases")],
    ]
    await update.message.reply_text(
        "👋 *Hola! Soy el bot de análisis de postura ergonómica.*\n\n"
        "Envíame una foto y analizaré tu postura al instante.\n\n"
        "¿Qué quieres saber?",
        reply_markup=InlineKeyboardMarkup(teclado),
        parse_mode="Markdown",
    )

async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ *¿Para qué sirve este bot?*\n\n"
        "Este bot analiza posturas ergonómicas usando inteligencia artificial.\n\n"
        "A partir de una foto, detecta los puntos clave de tu cuerpo (hombros, caderas, rodillas, etc.) "
        "y clasifica tu postura en 6 categorías:\n\n"
        "✅ *Correctas:* Bien de pie · Bien levantando · Bien sentado\n"
        "⚠️ *Incorrectas:* Mal de pie · Mal levantando · Mal sentado\n\n"
        "Además, mide los ángulos articulares reales y te da recomendaciones "
        "personalizadas según lo que encuentre en tu postura.\n\n"
        "🎯 *Objetivo:* Prevenir lesiones músculo-esqueléticas por malas posturas en el trabajo y la vida diaria.",
        parse_mode="Markdown",
    )

async def cmd_uso(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *¿Cómo usar el bot?*\n\n"
        "1️⃣ Toma una foto donde se vea tu cuerpo completo (o al menos desde la cabeza hasta las caderas).\n\n"
        "2️⃣ Envía la foto directamente a este chat.\n\n"
        "3️⃣ El bot responderá con:\n"
        "   • La foto con los puntos de detección dibujados\n"
        "   • La postura detectada y nivel de confianza\n"
        "   • Los ángulos articulares medidos\n"
        "   • Recomendaciones personalizadas\n\n"
        "📌 *Consejos para mejores resultados:*\n"
        "   • Foto de frente o de perfil lateral\n"
        "   • Iluminación adecuada\n"
        "   • Que se vea el cuerpo completo sin recortes\n"
        "   • Una sola persona en la foto\n\n"
        "💬 *Comandos disponibles:*\n"
        "   /start  — Menú principal\n"
        "   /info   — Información del bot\n"
        "   /uso    — Esta ayuda\n"
        "   /clases — Ver las 6 posturas que detecta",
        parse_mode="Markdown",
    )

async def cmd_clases(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎯 *Posturas que detecta el bot:*\n\n"
        "✅ *BIEN DE PIE*\n"
        "   Columna vertical, cabeza erguida, peso distribuido.\n\n"
        "✅ *BIEN LEVANTANDO*\n"
        "   Rodillas dobladas, espalda recta, carga cerca del cuerpo.\n\n"
        "✅ *BIEN SENTADO*\n"
        "   Espalda apoyada, caderas a 90°, pies en el suelo.\n\n"
        "⚠️ *MAL DE PIE*\n"
        "   Columna inclinada, cabeza adelantada, hombros caídos.\n\n"
        "⚠️ *MAL LEVANTANDO*\n"
        "   Espalda doblada, rodillas rectas al levantar peso.\n\n"
        "⚠️ *MAL SENTADO*\n"
        "   Espalda sin apoyo, cuello inclinado, postura encorvada.",
        parse_mode="Markdown",
    )

async def callback_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "info":
        await query.message.reply_text(
            "ℹ️ *¿Para qué sirve este bot?*\n\n"
            "Analiza posturas ergonómicas con IA. Detecta los puntos clave de tu cuerpo, "
            "mide ángulos articulares y clasifica tu postura en 6 categorías "
            "(3 correctas y 3 incorrectas), dando recomendaciones personalizadas.",
            parse_mode="Markdown",
        )
    elif query.data == "uso":
        await query.message.reply_text(
            "📖 Envía una foto directamente al chat.\nEl bot analizará tu postura y responderá "
            "con los puntos detectados, ángulos medidos y consejos personalizados.\n\n"
            "Escribe /uso para ver la guía completa.",
            parse_mode="Markdown",
        )
    elif query.data == "clases":
        await cmd_clases(query, context)

async def handle_foto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa cualquier foto enviada al bot."""
    msg = await update.message.reply_text("🔍 Analizando tu postura, un momento...")

    try:
        # Descargar la foto en mayor resolución disponible
        foto   = update.message.photo[-1]
        archivo = await foto.get_file()
        img_bytes = await archivo.download_as_bytearray()

        img_out, texto = analizar_imagen(bytes(img_bytes))

        if img_out:
            await update.message.reply_photo(
                photo=io.BytesIO(img_out),
                caption=texto,
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(texto, parse_mode="Markdown")

    except Exception as e:
        log.error(f"Error procesando foto: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Ocurrió un error al procesar la imagen. Intenta de nuevo."
        )
    finally:
        await msg.delete()

async def handle_documento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja fotos enviadas como archivo (documento)."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("Envía una imagen, no otro tipo de archivo.")
        return

    msg = await update.message.reply_text("🔍 Analizando tu postura, un momento...")
    try:
        archivo   = await doc.get_file()
        img_bytes = await archivo.download_as_bytearray()
        img_out, texto = analizar_imagen(bytes(img_bytes))

        if img_out:
            await update.message.reply_photo(
                photo=io.BytesIO(img_out),
                caption=texto,
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(texto, parse_mode="Markdown")
    except Exception as e:
        log.error(f"Error procesando documento: {e}", exc_info=True)
        await update.message.reply_text("❌ Error al procesar la imagen.")
    finally:
        await msg.delete()

async def handle_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Envíame una foto para analizar tu postura.\n"
        "Escribe /start para ver el menú o /uso para ver cómo usarme."
    )

# ─────────────────────────────────────────────
#  ARRANQUE CON RECONEXIÓN AUTOMÁTICA
# ─────────────────────────────────────────────

def main():
    log.info("Iniciando bot de posturas...")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("info",    cmd_info))
    app.add_handler(CommandHandler("uso",     cmd_uso))
    app.add_handler(CommandHandler("clases",  cmd_clases))
    app.add_handler(CallbackQueryHandler(callback_menu))
    app.add_handler(MessageHandler(filters.PHOTO,    handle_foto))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_documento))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_texto))

    # run_polling maneja reconexión automática ante caídas de red
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()
