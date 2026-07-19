# Presentación

`presentacion.md` es la presentación del TPF en formato **Marp** (Markdown → diapositivas). Tiene
14 diapositivas (portada + 12 de contenido + cierre); el bloque de **métricas centradas en eventos**
(diapositivas 8–10) es el núcleo del aporte del trabajo.

## Cómo exportar a PDF o PowerPoint

**Opción A — Extensión de VS Code (más simple):**
1. Instalar la extensión *Marp for VS Code*.
2. Abrir `presentacion.md`.
3. Botón *Export slide deck…* → elegir PDF o PPTX.

**Opción B — Línea de comandos (Marp CLI):**
```bash
# Requiere Node.js
npx @marp-cli/marp-cli presentacion.md --pdf     # PDF
npx @marp-cli/marp-cli presentacion.md --pptx    # PowerPoint
```

Las figuras se referencian con rutas relativas a `../results/figures/`, por lo que la exportación
debe hacerse desde esta carpeta (`presentacion/`) o desde la raíz del repositorio.
