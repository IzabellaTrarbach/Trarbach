# Prova 2 de Controle de Processos: circuito RC de 3ª ordem (circuito G, Vc3/Vin)

- `entrega/` : documento a entregar (LaTeX em `main.tex` + `parte1..4.tex`, PDF `Resolucao_Prova2_entrega.pdf`),
  com cálculos por extenso, resultados experimentais da Questão 1 (Q11, Q12, Q13a, Q13b), dados (`dados/`),
  figuras (`figs/`, `figs_exp/`) e códigos de comparação (`codigos/`).
- `guia/` : guia de estudo e roteiro de bancada (LaTeX em `guia.tex`, PDF `Guia_estudo_e_bancada.pdf`) e scripts.

Compilar: `cd entrega && pdflatex main.tex` (3x) ; `cd guia && pdflatex guia.tex` (3x).
Reproduzir a análise: `cd entrega && python3 codigos/analise_experimentos.py` (numpy, scipy, matplotlib).
