## Análise Crítica e Fundamentação

### Escalonamento e Distâncias — por que o `StandardScaler` é indispensável para o KNN nesta base?

O `KNeighborsClassifier` classifica uma amostra com base na **distância** (por padrão, Euclidiana) até os vizinhos mais próximos. A distância Euclidiana soma as diferenças **ao quadrado** de cada atributo:

$$d(x, x') = \sqrt{\sum_{i=1}^{13} (x_i - x'_i)^2}$$

No dataset Wine, os 13 atributos têm escalas muito diferentes. Por exemplo:

- `proline` varia na casa das **centenas/milhares** (aprox. 278–1687);
- `magnesium` varia em torno de **dezenas/centenas** (aprox. 70–162);
- `hue`, `od280/od315_of_diluted_wines` e `malic_acid` variam em frações ou dezenas pequenas.

Sem escalonamento, atributos com magnitude maior (como `proline`) **dominam completamente** o cálculo da distância, e o KNN passa a "ignorar" atributos com valores pequenos, mesmo que sejam informativos. Isso degrada a acurácia e torna o modelo sensível à unidade de medida.

O `StandardScaler` padroniza cada atributo para **média 0 e desvio padrão 1**:

$$z = \frac{x - \mu}{\sigma}$$

Assim, todos os atributos passam a contribuir de forma equilibrada para a distância. É por isso que o escalonamento é **indispensável** para o KNN nesta base — sem ele, o modelo essencialmente mede a distância em uma única dimensão (a de maior escala).

> Observação: a Regressão Logística também se beneficia do escalonamento (convergência mais rápida e regularização mais justa), mas o efeito é crítico para o KNN por causa da natureza baseada em distância.

### 5.2 Prevenção de vazamento — como o `Pipeline` evita data leakage no `cross_val_score`?

O vazamento de dados (*data leakage*) ocorre quando informações do conjunto de teste (ou de validação) influenciam o treinamento. Um exemplo clássico e sutil é **ajustar o `StandardScaler` em todos os dados antes da validação cruzada**: nesse caso, a média e o desvio padrão usados no escalonamento já "viram" os dados de validação, contaminando o processo.

O `Pipeline` resolve isso porque o `cross_val_score`, a cada um dos 5 folds:

1. **Divide** os dados de treino em sub-treino e sub-validação;
2. **Ajusta (`fit`) o `StandardScaler` apenas no sub-treino** daquele fold;
3. **Transforma** o sub-treino e o sub-validação com os parâmetros aprendidos **somente no sub-treino**;
4. Treina o classificador no sub-treino transformado e avalia no sub-validação transformado;
5. Repete para os demais folds, sempre reajustando o scaler do zero.

Ou seja, o `Pipeline` garante que **toda etapa de pré-processamento seja tratada como parte do modelo** e reajustada a cada fold. Nenhuma estatística (média, desvio padrão, etc.) do conjunto de validação ou de teste é usada para treinar o pipeline. O mesmo vale para o `fit` final no conjunto de treino completo e o `predict` no teste: o scaler é ajustado apenas no treino e apenas **aplicado** (`transform`) no teste.

Em resumo: encapsular o `StandardScaler` no `Pipeline` **isola o pré-processamento dentro do ciclo de validação**, impedindo que informação dos dados de validação/teste "vaze" para o treinamento — condição essencial para uma estimativa honesta de desempenho.
