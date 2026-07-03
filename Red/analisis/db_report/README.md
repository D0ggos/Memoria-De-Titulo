# Análisis de la base de datos RS

Caracterización tipo *data science* de `DB_ssf_RS_500_c.mat` para la tesis.

## Contenido
- `analisis_db.tex` — sección de tesis (LaTeX, `\input`). Requiere `graphicx`,
  `booktabs`, `amsmath`, `amssymb` y `\graphicspath{{figuras/}}`.
- `figuras/fig1..fig8.pdf` — figuras vectoriales.
- `CONTEXTO_base_de_datos.md` — resumen compacto para pegar como contexto a un chat.
- `descriptors.csv` — un descriptor por sistema (14 000 filas).
- `arrays.npz` — autovalores de lazo abierto/cerrado (para histogramas).
- `stats.json` — números citados en el texto.

## Reproducir
```bash
../../.venv/bin/python extract_db.py   # .mat -> descriptors.csv + arrays.npz
../../.venv/bin/python figs.py         # -> figuras/*.pdf
```
(Ejecutar desde esta carpeta. El `.mat` se lee de `../../DB_ssf_RS_500_c.mat`.)
