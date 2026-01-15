# LLM-and-Ontologies

Este repositório contém os códigos e arquivos de uma pesquisa voltada para o enriquecimento semântico de regras de associação em dados de IoT.
![arquitetura](https://github.com/user-attachments/assets/32dac771-21d6-4391-8c6b-e1a0c1db64a5)


## 📖 Sobre a Pesquisa

O projeto implementa um pipeline que integra Mineração de Regras de Associação (ARM), Grandes Modelos de Linguagem (LLMs) e Ontologias. Conforme ilustrado na arquitetura do projeto, o fluxo de trabalho consiste em:

1.  **Mineração de Dados IoT:** Processamento de dados brutos para extração de padrões.
2.  **Geração de Regras Candidatas:** Utilização de algoritmos ARM para identificar regras preliminares.
3.  **Enriquecimento Semântico (LLMs + Ontologias):** As regras candidatas são processadas por LLMs que realizam *Entity Linking* (vinculação de entidades) e consultam uma ontologia específica.
4.  **Grafo de Conhecimento:** Geração e incorporação de triplas semânticas para formar um grafo de conhecimento.
5.  **Avaliação:** Análise final das regras resultantes enriquecidas.

## 🚀 Como Executar

Siga a ordem abaixo para preparar o ambiente e rodar os experimentos.

### 1. Pré-requisitos (Dataset)

Antes de iniciar a execução dos scripts, é necessário baixar o conjunto de dados **CACHET-CADB**.

* **Download:** [https://data.dtu.dk/articles/dataset/CACHET-CADB/14547264](https://data.dtu.dk/articles/dataset/CACHET-CADB/14547264)
* Certifique-se de extrair os arquivos e organizá-los na estrutura de pastas correta do projeto.

### 2. Processamento dos Dados

O fluxo de execução dos scripts Python deve seguir estritamente a ordem abaixo:

**Passo 1: Gerar CSVs Auxiliares**
Execute o script `gerar-csv-auxiliares.py`.
> Este script percorre todas as pastas do dataset baixado para consolidar e gerar os arquivos CSV auxiliares necessários para as etapas seguintes.

```bash
python gerar-csv-auxiliares.py
```

**Passo 2: Gerar Contexto de Arritmias**
Execute o script `gerar-csv-contexto-arritmias.py`.
> Este passo prepara os dados focados no contexto específico das arritmias cardíacas.

```bash
python gerar-csv-contexto-arritmias.py
```

**Passo 3: Algoritmo Genético e Apriori**
Por fim, execute o arquivo `genetic-algorithm.py`.
> Este script utiliza o CSV de contexto das arritmias gerado anteriormente. Ele aplica um Algoritmo Genético em conjunto com a técnica Apriori para gerar as regras de associação finais.
```bash
python genetic-algorithm.py
```

## ⚠️ Estado do Desenvolvimento
Este repositório ainda está em fase de desenvolvimento. Portanto:

- Podem ocorrer falhas durante a execução de alguns arquivos.
- O código está sujeito a alterações e otimizações.
