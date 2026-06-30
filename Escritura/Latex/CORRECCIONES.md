# Correcciones aplicadas a la Memoria de Título

Todo el texto agregado por la revisión está en **azul** (`\rev{}` o `{\color{revcolor} ...}`).
Las notas de revisión van como `\revnota{}` en azul y negrita. Los `\pendiente{}` rojos del autor
se conservan solo donde no se pudieron resolver sin tu criterio.

## Arreglos de LaTeX
1. **Referencias de ecuaciones (crítico):** las ecuaciones se citaban a mano ("Ecuación (8)", "(3)",
   "(2)", "(4)", "(7)", "(9)", "(1)") y no había ningún `\label`. En `report` se numeran como (2.x),
   así que TODAS estaban mal. Se agregaron `\label{}` a las 8 ecuaciones clave y se reemplazaron las
   referencias por `\eqref{}`:
   - `eq:vdot_cuadratica`, `eq:lyap_lpv_infinita`, `eq:dinamica_lazo_cerrado`,
     `eq:bmi_expandida`, `eq:congruencia_KQ`, `eq:lmi_sintesis`, `eq:proyeccion`.
2. **Anexo faltante:** `\include{Capitulos/Anexo1}` apuntaba a un archivo inexistente. Se creó
   `Capitulos/Anexo1.tex` (documentación del dataset e hiperparámetros).
3. **Librerías TikZ:** se agregó `\usetikzlibrary{positioning,calc}` (faltaban para la figura nueva).

## `\pendiente` resueltos (en azul)
- **Introducción:** casos reales de CPS mal estabilizados (apagón 2003, ataques a CPS) + citas.
- **Planteamiento:** origen de la base de datos (grupo Oliveira & Peres, UNICAMP) + citas.
- **Marco teórico:** "cuál ecuación?" (x2) resuelto vía `\eqref`; figura de trayectoria → ref a
  `fig:politopo_limites`; citas de polítopos y de offline-online/ML agregadas; figura de
  realimentación de estados creada (`fig:realimentacion`, TikZ); referencia a DR convertida en
  cross-reference a la sección de la LMI-Net; dos notas "revisar sección" convertidas en `\revnota`.

## Contenido nuevo (en azul, para que lo edites)
- **conclusiones.tex** (estaba casi vacío): sección de Trabajo Futuro con
  (a) algoritmos alternativos a Douglas–Rachford y (b) nuevos criterios de desempeño.

## Entradas nuevas en biblio.bib
`boyd1994lmi`, `oliveira2008robust`, `robustdatabase_unicamp`, `uscanada2004blackout`,
`boyle1986dykstra`, `bauschke1994dykstra`, `bai2019deq`, `gould2021ddn`, `boyd2011admm`.
> Verifica/ajusta los datos bibliográficos exactos (volumen, páginas, año) antes de entregar.

## Compilación
Compila limpio con `pdflatex → bibtex → pdflatex → pdflatex`: 46 páginas, 0 referencias rotas,
0 citas indefinidas. Ver `MT_revisado.pdf`.
