## Método

### Dados e alvo
- **Apólices:** PSR/SISSER (MAPA), uma linha por apólice de seguro rural subvencionada,
  2006-2025. Nome e CPF/CNPJ são descartados de cada bloco enquanto o CSV é lido em
  streaming, antes de qualquer gravação. Depois da limpeza ficam 1.532.459 apólices
  ([DATA_CARD](DATA_CARD.md)).
- **Alvo:** apólice indenizada (`VALOR_INDENIZAÇÃO > 0`).
- **Censura:** os arquivos públicos param de registrar sinistros no início de 2025
  (apólices que começam em dezembro de 2024 têm 0,09% de sinistro; o arquivo de 2025
  não tem nenhum). Uma safra só entra se 98% ou mais das apólices terminaram a
  janela de risco 90 dias antes da data de corte estimada. A 2024/25 fica fora; a
  última safra de teste é a 2023/24.
- **Um problema na fonte, encontrado e tratado:** o arquivo 2016-2024 foi cortado no
  limite de linhas do Excel (1.048.565 linhas); a seguradora no fim do arquivo perde
  as linhas de 2020 a 2023 e por isso sai a partir de 2020 ([DECISIONS D2](DECISIONS.md)).

### Quando o clima importa? Um calendário agrícola
As datas de vigência do PSR são contratuais: a cobertura da soja costuma começar
meses antes do plantio e durar 365 dias. Por isso cada grupo de culturas tem uma
**janela crítica** tirada dos calendários da CONAB (soja no Sul: 1º/dez + 105 dias;
milho safrinha: 15/mar + 107 dias; trigo no Sul: 1º/jul + 122 dias), e cada apólice
recebe a primeira janela depois da contratação. Nas apólices de 2016 em diante, a
janela cai dentro da vigência real em 99% dos casos de soja e milho safrinha.

A safra (o fold) é o ano agosto-julho do início da janela: soja de verão, o milho
safrinha seguinte e o trigo de inverno da mesma temporada ficam no mesmo fold.

### Variáveis
| Bloco | O que é | Conhecido quando |
|---|---|---|
| Contrato | cultura, UF, seguradora, tipo de produto, área, nível de cobertura, soma segurada/ha e produtividade esperada relativas à mediana da safra anterior, mês da contratação, dias até a janela crítica | contratação |
| Histórico da carteira | taxa de sinistro passada do município x grupo de cultura, UF x grupo, grupo (só safras anteriores, com encolhimento para o nível acima, n >= 10) | contratação |
| Climatologia | estatísticas 1981-2005 da janela crítica da própria apólice na célula do NASA POWER: chuva normal, variabilidade, frequência de seca, veranicos, chuva máxima em 5 dias, dias quentes, dias com risco de geada, umidade do solo | contratação |
| ENSO | último Índice Oceânico do Niño publicado e sua variação em 3 meses | contratação |
| Clima da safra | **anomalias** de chuva, veranico, calor, geada e umidade do solo do início da janela até a metade, mais os 30 dias anteriores | meio da janela |

### Validação
- **Walk-forward por safra:** para cada safra de teste de 2013/14 a 2023/24, treina
  em todas as safras anteriores e avalia na safra de teste (11 folds). Nunca split
  aleatório.
- **Hiperparâmetros** escolhidos uma vez, nas safras 2009-2012 (antes de qualquer
  safra de teste), pela média de AUC nesses anos.
- **Auditoria de vazamento:** indenização, evento causador, prêmio e taxa nunca são
  variáveis dos modelos principais (testado); o histórico é calculado "as-of"; o
  período da climatologia termina antes da primeira apólice (testado); as variáveis
  da safra param na metade da janela crítica.
- **Sensibilidade:** um intervalo de uma safra entre treino e teste (os sinistros da
  última safra ainda podem estar abertos quando a próxima é precificada) e janela
  móvel de 8 safras.
