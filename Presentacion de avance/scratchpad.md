# Scratchpad — Deck Avance MT (Pedro Muñoz, UdeC)

## Title sequence (topic noun-phrase style, español)
Dividers in CAPS-roman-numeral, dark navy.

01  Portada (no title bar)
02  Sistemas ciberfísicos y la exigencia de estabilidad
03  Sistemas LPV politópicos y la barrera computacional
04  Pregunta de investigación  [dark, centered]
—   I · MARCO TEÓRICO  [divider]
05  Lyapunov: energía virtual y disipación
06  De BMI a LMI: síntesis por congruencia
07  Proyección sobre la intersección de convexos
08  Estado del arte: restricciones duras en redes neuronales
09  Evolución de la idea: de la propuesta al estado actual
10  Objetivo general
11  Objetivos específicos
—   II · DISEÑO DE LA SOLUCIÓN  [divider]
12  La descomposición offline–online
13  Arquitectura LMI-Net
14  La capa LMI-Net en detalle
15  Criterios de desempeño y entrenamiento auto-supervisado
—   III · DESARROLLO Y RESULTADOS  [divider]
16  Metodología y dataset UNICAMP
17  Resultados: estabilización
18  Resultados: velocidad de inferencia
—   IV · ANÁLISIS DEL PROYECTO  [divider]
19  Línea en curso: rigidez dimensional
20  Análisis de riesgos
21  Restricciones del proyecto
22  Condiciones para operación adecuada
23  Costos y beneficios
24  Conclusiones del avance y trabajo futuro
25  Cierre

Total = 25 brief slides + 4 dividers = 29 sections.

## Design system
Fonts: IBM Plex Sans (titles/body), IBM Plex Mono (diagrams/features/code), MathJax for equations.
Colors:
  --blue   #003A70  primary institutional
  --blue2  #00529B  brighter accent/link
  --navy   #07223D  dark slide bg
  --navy2  #0C2C4C  dark panel
  --bg     #F5F4F0  warm off-white content bg
  --panel  #FFFFFF / #ECEBE5 panels
  --ink    #16202E  text
  --muted  #5B6573  secondary text
  --cyan   #0E8FA8  neutral concept highlight
  --green  #1E8E5A  positive result
  --amber  #C2890F  risk/limitation
  --red    #B23A2E  hard limitation
  --gray   #6E7681  table/baseline
Semantics: green=positive results, amber/red=risk/limit, cyan=neutral concept, gray=baseline.

Type scale @1920: title 58, subtitle 38, body 30, small 26, micro 24 (floor).
Pad: top 84, bottom 68, x 92. gap-title 36, gap-item 22.

## Layout system
- Eyebrow (block label + slide kicker) top-left, mono, cyan/muted.
- Title h2 below eyebrow.
- Thin rule under title (blue).
- Content area flexible.
- Footer: running footer w/ author + slide nb optional (skip — deck overlay handles count).
- Dividers: full navy, big roman numeral + block title, small running list.
- Equations: MathJax, sized up.
- Tables: clean, hairline rows, header in blue, semantic color cells.
