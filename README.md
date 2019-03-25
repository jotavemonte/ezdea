# Ezdea promove facilidade ao usar DEA com Python

## Como usar:

```python
# crie a situação a ser analisada. No nosso caso será a seguinte matriz:

matriz = [[20, 151, 100, 90],
          [19, 131, 150, 50],
          [25, 160, 160, 55],
          [27, 168, 180, 72],
          [22, 158, 94, 66],
          [55, 255, 230, 90],
          [33, 235, 220, 88],
          [31, 206, 152, 80],
          [30, 244, 290, 100],
          [50, 268, 250, 100],
          [53, 306, 262, 147],
          [38, 284, 250, 120]]
          
# declare as entradas e saídas do problema
numero_inputs = 2
numero_outputs = 2

# instancie o cenário
cenario = DEA(matriz, numero_inputs, numero_outputs)
resultado_bcc = cenario.bcc_dual_input(matrix=True) # matrix = True força que o retorno seja matriz e não json
df_bcc_dual = pd.DataFrame(resultado_bcc)
print(df_bcc_dual)
metas = cenario.calcular_metas(resultado_bcc)
df_metas = pd.DataFrame(metas)
print(df_metas)

```

## Lista de métodos:

* DEA.ccr_primal_input(matrix=False)

* DEA.ccr_primal_output(matrix=False)

* DEA.ccr_dual_input(matrix=False)

* DEA.ccr_dual_output(matrix=False)

* DEA.bcc_primal_input(matrix=False)

* DEA.bcc_primal_output(matrix=False)

* DEA.bcc_dual_input(matrix=False)

* DEA.bcc_dual_output(matrix=False)

* DEA.calcular_metas(matriz_dual)
