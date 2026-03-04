import random
import time
import pandas as pd
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules

# =============================================================================
# NOVO PRÉ-PROCESSAMENTO
# =============================================================================

def pre_processamento_completo():
    df = pd.read_csv("generated-csv\contexto_das_arritmias.csv")

    # =============================
    # IDENTIFICAÇÃO
    # =============================
    df['patient_id'] = df['patient']
    df['ecg_id'] = (
        df['patient'].astype(str) + '_' +
        df['session'].astype(str) + '_' +
        df['segment'].astype(str) + '_' +
        df['Start'].astype(str) + '_' +
        df['End'].astype(str)
    )
    df['rr_interval'] = df['Start'].astype(str) + ' + ' + df['End'].astype(str)
    df['heart_rate'] = pd.to_numeric(df['Hr [1/min]'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['Date abs [yyyy-mm-dd]'], errors='coerce')

    # =============================
    # MAPEAMENTOS
    # =============================
    activity_map = {
        0: "activity_unknown",
        1: "activity_lying",
        2: "activity_sitting_standing",
        3: "activity_cycling",
        4: "activity_slope_up",
        5: "activity_jogging",
        6: "activity_slope_down",
        7: "activity_walking",
        8: "activity_sitting_lying",
        9: "activity_standing",
        10: "activity_sitting_lying_standing",
        11: "activity_sitting",
        99: "activity_not_worn"
    }
    bodypos_map = {
        0: "body_unknown",
        1: "body_lying_supine",
        2: "body_lying_left",
        3: "body_lying_prone",
        4: "body_lying_right",
        5: "body_upright",
        6: "body_sitting_lying",
        7: "body_standing",
        99: "body_not_worn"
    }
    sleepwake_map = {
        0: "wake",
        1: "sleep",
        2: "not_worn"
    }
    gender_map = {
        "M": "gender_M", "F": "gender_F",
        "m": "gender_M", "f": "gender_F"
    }
    class_map = {
        1: "AF (Atrial Fibrillation)",
        2: "NSR (Normal Sinus Rhythm)",
        3: "Noise",
        4: "Others"
    }

    df['ActivityClass_mapped'] = pd.to_numeric(
        df['ActivityClass []'], errors='coerce').map(activity_map)
    df['BodyPosition_mapped'] = pd.to_numeric(
        df['BodyPosition []'], errors='coerce').map(bodypos_map)
    df['NonWearSleepWake_mapped'] = pd.to_numeric(
        df['NonWearSleepWake []'], errors='coerce').map(sleepwake_map)
    df['gender_mapped'] = df['gender'].astype(str).str.strip().map(gender_map)
    df['ArrhythmiaClass'] = pd.to_numeric(
        df['Class'], errors='coerce').map(class_map)

    # =============================
    # HEART RATE EM 4 FAIXAS
    # =============================
    df['heart_rate_bin'] = pd.cut(
        df['heart_rate'],
        bins=[0, 60, 90, 120, df['heart_rate'].max()],
        labels=["hr_bradycardia", "hr_normal", "hr_elevated", "hr_tachycardia"],
        include_lowest=True
    )

    # =============================
    # BINNING
    # =============================
    df["MET_bin"] = pd.cut(
        df["MET []"],
        bins=[0, 1.0, 1.25, df["MET []"].max()],
        labels=["met_1.0", "met_1.25", "met_acima_1.25"],
        include_lowest=True
    )
    df["acc_bin"] = pd.cut(
        df["MovementAcceleration [g]"],
        bins=[0, 0.004326, 0.007704, 0.021811, df["MovementAcceleration [g]"].max()],
        labels=["acc_muito_baixa", "acc_baixa", "acc_moderada", "acc_alta"],
        include_lowest=True
    )
    df["weight_bin"] = pd.cut(
        df["weight"],
        bins=[50, 70, 79, 86, df["weight"].max()],
        labels=["peso_muito_baixo", "peso_baixo", "peso_medio", "peso_alto"],
        include_lowest=True
    )
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[30, 40, 50, 60, 80],
        labels=["adulto_jovem", "adulto", "quase_idoso", "idoso"],
        include_lowest=True
    )
    df["height_bin"] = pd.cut(
        df["height"],
        bins=[75, 102.5, 130, 157.5, 185],
        labels=["height_baixo", "height_medio", "height_alto", "height_muito_alto"],
        include_lowest=True
    )

    # =============================
    # DATAFRAME FINAL (SEM COLUNA CONTEXT)
    # =============================
    df_final = df[
        [
            "patient_id",
            "ecg_id",
            "rr_interval",
            "timestamp",
            # CONTEXTO (como colunas) — usadas exclusivamente pelo algoritmo genético
            "heart_rate_bin",
            "ActivityClass_mapped",
            "BodyPosition_mapped",
            "NonWearSleepWake_mapped",
            "gender_mapped",
            "MET_bin",
            "acc_bin",
            "weight_bin",
            "age_bin",
            "height_bin",
            "ArrhythmiaClass"
        ]
    ]
    return df_final, df  # retorna também o df original com todas as colunas


# =============================================================================
# COLUNAS DE CONTEXTO USADAS PELO ALGORITMO GENÉTICO
# =============================================================================

COLS_CONTEXTO = [
    # CONTEXTO (como colunas)
    "heart_rate_bin",
    "ActivityClass_mapped",
    "BodyPosition_mapped",
    "NonWearSleepWake_mapped",
    "gender_mapped",
    "MET_bin",
    "acc_bin",
    "weight_bin",
    "age_bin",
    "height_bin",
    "ArrhythmiaClass"
]


def preparar_dados_ga(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para o algoritmo genético usando APENAS as colunas de contexto.
    Retorna o dataframe one-hot encoded (booleano) para uso no Apriori.
    """
    df_cat = df_final[COLS_CONTEXTO].dropna().copy()
    df_hot = pd.get_dummies(df_cat, prefix_sep='=')
    df_hot_bool = (df_hot > 0).astype(bool)
    return df_hot_bool


def aplicar_regras_ao_dataframe(df_final: pd.DataFrame, rules: pd.DataFrame) -> list:
    """
    Para cada linha do df_final, verifica quais regras (antecedentes) são satisfeitas.
    Retorna uma lista com a regra de maior lift que se aplica a cada linha,
    ou None caso nenhuma regra se aplique.
    """
    def linha_para_itens(row):
        """Converte uma linha em conjunto de strings 'coluna=valor'."""
        itens = set()
        for col in COLS_CONTEXTO:
            val = row.get(col)
            if pd.notna(val):
                itens.add(f"{col}={val}")
        return itens

    def regra_aplicavel(itens_linha: set, antecedentes: frozenset) -> bool:
        """Verifica se todos os antecedentes da regra estão presentes na linha."""
        return set(antecedentes).issubset(itens_linha)

    regras_por_linha = []

    for _, row in df_final.iterrows():
        itens_linha = linha_para_itens(row)
        melhor_regra_str = None
        melhor_lift = -1

        for _, rule in rules.iterrows():
            if regra_aplicavel(itens_linha, rule['antecedents']):
                if rule['lift'] > melhor_lift:
                    melhor_lift = rule['lift']
                    ant_str = " AND ".join(sorted(str(i) for i in rule['antecedents']))
                    cons_str = " AND ".join(sorted(str(i) for i in rule['consequents']))
                    melhor_regra_str = f"{ant_str} => {cons_str}"

        regras_por_linha.append(melhor_regra_str)

    return regras_por_linha


# =============================================================================
# DADOS GLOBAIS — inicializados após o pré-processamento
# =============================================================================

print("▶ Executando pré-processamento...")
DF_FINAL, DF_ORIGINAL_COMPLETO = pre_processamento_completo()
DADOS = preparar_dados_ga(DF_FINAL)
print(f"✔ Pré-processamento concluído. Shape para o GA: {DADOS.shape}")


# =============================================================================
# ALGORITMO GENÉTICO
# =============================================================================

class Individuo:
    def __init__(self, min_support=None, max_len=None):
        if min_support is None and max_len is None:
            self.min_support = round(random.uniform(0.01, 0.50), 4)
            self.max_len = random.randint(2, 7)
        else:
            self.min_support = min_support
            self.max_len = max_len

        self.rules = None
        self.fitness_score = self.calcular_fitness()

    def calcular_fitness(self, max_tentativas: int = 5) -> float:
        """
        Calcula o fitness do indivíduo.

        Causas conhecidas de NaN:
          - apriori não encontra itemsets com o min_support atual  →  DataFrame vazio
          - filter_rules descarta todas as regras geradas           →  DataFrame vazio
          - .mean() em série vazia retorna NaN                      →  fitness = NaN

        Estratégia de correção:
          1. Se o resultado for NaN, o indivíduo é re-sorteado e reavaliado.
          2. Após `max_tentativas` sem sucesso, retorna fitness = 0.0 como fallback
             seguro, evitando que NaN se propague pelo algoritmo genético.
        """
        for tentativa in range(1, max_tentativas + 1):
            frequent_itemsets = apriori(
                DADOS,
                min_support=self.min_support,
                use_colnames=True,
                max_len=self.max_len,
            )

            # Apriori pode retornar DataFrame vazio se nenhum itemset atingir o suporte
            if frequent_itemsets.empty:
                self._resorteiar_parametros()
                continue

            raw_rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=0.4
            )
            filtered = self.filter_rules(raw_rules)

            # filter_rules pode descartar todas as regras (sem ArrhythmiaClass)
            if filtered.empty:
                self._resorteiar_parametros()
                continue

            mean_lift    = filtered['lift'].mean()
            mean_support = filtered['support'].mean()

            # Verificação explícita de NaN antes de aceitar o resultado
            if pd.isna(mean_lift) or pd.isna(mean_support):
                self._resorteiar_parametros()
                continue

            # Resultado válido — armazena e retorna
            self.rules = filtered
            return float(mean_lift * mean_support)

        # Fallback: nenhuma tentativa produziu resultado válido
        print(
            f"  ⚠ Indivíduo não convergiu após {max_tentativas} tentativas "
            f"(min_support={self.min_support}, max_len={self.max_len}). "
            f"Fitness definido como 0.0."
        )
        if self.rules is None:
            # Garante que self.rules nunca seja None para evitar erros posteriores
            self.rules = pd.DataFrame(
                columns=['antecedents', 'consequents', 'support',
                         'confidence', 'lift', 'leverage', 'conviction']
            )
        return 0.0

    def _resorteiar_parametros(self):
        """Re-sorteia min_support e max_len para uma nova tentativa de avaliação."""
        self.min_support = round(random.uniform(0.01, 0.50), 4)
        self.max_len     = random.randint(2, 7)

    def filter_rules(self, temporary_rules):
        """Mantém apenas regras que envolvam ArrhythmiaClass em antecedentes ou consequentes."""
        new_rules = temporary_rules[
            temporary_rules['antecedents'].apply(
                lambda items: any(str(i).startswith("ArrhythmiaClass=") for i in items)
            ) |
            temporary_rules['consequents'].apply(
                lambda items: any(str(i).startswith("ArrhythmiaClass=") for i in items)
            )
        ].copy()
        return new_rules

    def __str__(self):
        if self.rules is not None and not self.rules.empty:
            lift_str    = f"{self.rules['lift'].mean():.4f}"
            support_str = f"{self.rules['support'].mean():.4f}"
        else:
            lift_str    = "N/A (sem regras validas)"
            support_str = "N/A (sem regras validas)"
        return (
            f"Individuo(min_support={self.min_support}, max_len={self.max_len}, "
            f"media do lift={lift_str}, "
            f"media do support={support_str}, "
            f"fitness={self.fitness_score:.4f})"
        )


class GA:
    def __init__(self, individuo: int, geracao: int, mutacao: int):
        self.individuo = individuo
        self.geracao = geracao
        self.mutacao = mutacao
        self.melhor_individuo = None
        self.executar()

    def executar(self):
        self.populacao_atual = self.gerarPopulacao()
        self.melhor_individuo = self.acharMelhorIndividuo()
        media_fitness_geracao = []

        somat = sum(ind.fitness_score for ind in self.populacao_atual)
        media_fitness_geracao.append(somat / len(self.populacao_atual))

        for _ in tqdm(range(self.geracao - 1)):
            self.crossover()
            self.mutar()
            self.acharMelhorIndividuo()

            soma = 0
            for ind in self.populacao_atual:
                ind.calcular_fitness()
                soma += ind.fitness_score
            media_fitness_geracao.append(soma / len(self.populacao_atual))

        self.guardar_resultado()

        for i, media in enumerate(media_fitness_geracao):
            print(f"Geração {i + 1}: Média do fitness = {media:.4f}")

    def gerarPopulacao(self) -> list:
        return [Individuo() for _ in range(self.individuo)]

    def acharMelhorIndividuo(self) -> Individuo:
        melhor = max(self.populacao_atual, key=lambda ind: ind.fitness_score)
        if self.melhor_individuo is None or melhor.fitness_score > self.melhor_individuo.fitness_score:
            self.melhor_individuo = melhor
        return self.melhor_individuo

    def crossover(self):
        nova_geracao = []
        for _ in range(self.individuo // 2):
            x = self.selecao()
            y = self.selecao()
            nova_geracao.append(Individuo(min_support=x.min_support, max_len=y.max_len))
            nova_geracao.append(Individuo(min_support=y.min_support, max_len=x.max_len))
        self.populacao_atual = nova_geracao

    def mutar(self):
        for individuo in self.populacao_atual:
            if random.randint(1, 100) <= self.mutacao:
                individuo.min_support = round(random.uniform(0.01, 0.50), 4)
                individuo.max_len = random.randint(2, 6)
                individuo.fitness_score = individuo.calcular_fitness()

    def selecao(self) -> Individuo:
        x, y = random.sample(range(len(self.populacao_atual)), 2)
        if self.populacao_atual[x].fitness_score > self.populacao_atual[y].fitness_score:
            return self.populacao_atual[x]
        return self.populacao_atual[y]

    def guardar_resultado(self):
        print("\n✔ Melhor indivíduo encontrado:", self.melhor_individuo)

        # -----------------------------------------------------------------
        # 1. Salvar as regras geradas pelo GA (arquivo auxiliar)
        # -----------------------------------------------------------------
        caminho_regras = "resultado_GA.csv"
        try:
            self.melhor_individuo.rules.to_csv(caminho_regras, index=False, encoding='utf-8')
            print(f"✔ Regras do GA salvas em: {caminho_regras}")
        except Exception as e:
            print("❌ Erro ao salvar regras:", e)

        # -----------------------------------------------------------------
        # 2. Aplicar regras ao DF_FINAL e gerar CSV com estrutura original
        #    + nova coluna 'regra_ga'
        # -----------------------------------------------------------------
        print("▶ Aplicando regras ao dataset original...")
        regras_por_linha = aplicar_regras_ao_dataframe(DF_FINAL, self.melhor_individuo.rules)

        # Cria cópia do dataframe completo (todas as colunas originais)
        df_saida = DF_ORIGINAL_COMPLETO.copy()
        # Garante alinhamento de índices com DF_FINAL
        df_saida = df_saida.loc[DF_FINAL.index].copy()
        df_saida['regra_ga'] = regras_por_linha

        caminho_saida = "resultado_com_regras.csv"
        try:
            df_saida.to_csv(caminho_saida, index=False, encoding='utf-8')
            print(f"✔ Dataset completo com regras salvo em: {caminho_saida}")
            total = len(df_saida)
            com_regra = df_saida['regra_ga'].notna().sum()
            print(f"   → {com_regra}/{total} linhas possuem regra associada "
                  f"({100 * com_regra / total:.1f}%)")
        except Exception as e:
            print("❌ Erro ao salvar dataset com regras:", e)


# =============================================================================
# EXECUÇÃO
# =============================================================================

t1 = time.perf_counter()
ga = GA(individuo=20, geracao=100, mutacao=20)
t2 = time.perf_counter()
print(f"\nTempo de execução: {(t2 - t1):.2f}s")