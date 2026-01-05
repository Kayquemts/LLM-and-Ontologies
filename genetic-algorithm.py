import random
import pandas as pd
from sklearn import metrics 
from tqdm import tqdm
from mlxtend.frequent_patterns import apriori, association_rules    


def pre_processamento():
    df = pd.read_csv(r'generated-csv\contexto_das_arritmias.csv')

    manter = [
        'ActivityClass []', 'BodyPosition []', 'MET []',
        'MovementAcceleration [g]', 'NonWearSleepWake []',
        'gender', 'weight', 'age', 'height', 'Class'
    ]
    df = df[manter].copy()
    df = df.dropna().copy()

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
        "M": "gender_M",
        "F": "gender_F",
        "m": "gender_M",
        "f": "gender_F"
    }

    class_map = {
        1: "AF (Atrial Fibrillation)",
        2: "NSR (Normal Sinus Rhythm)",
        3: "Noise",
        4: "Others"
    }

    df['ActivityClass_mapped'] = pd.to_numeric(df['ActivityClass []'], errors='coerce').map(activity_map)
    df['BodyPosition_mapped'] = pd.to_numeric(df['BodyPosition []'], errors='coerce').map(bodypos_map)
    df['NonWearSleepWake_mapped'] = pd.to_numeric(df['NonWearSleepWake []'], errors='coerce').map(sleepwake_map)
    df['ArrhythmiaClass'] = pd.to_numeric(df['Class'], errors='coerce').map(class_map)
    df['gender_mapped'] = df['gender'].astype(str).map(lambda x: x.strip()).map(gender_map)

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

    cols_apriori = [
        'ActivityClass_mapped',
        'BodyPosition_mapped',
        'NonWearSleepWake_mapped',
        'gender_mapped',
        'MET_bin',
        'acc_bin',
        'weight_bin',
        'age_bin',
        'height_bin',
        'ArrhythmiaClass'
    ]

    df_cat = df[cols_apriori].copy()

    df_hot = pd.get_dummies(df_cat, prefix_sep='=')
    df_hot_bool = (df_hot > 0).astype(bool)

    return df_hot_bool

DADOS = pre_processamento()

class Individuo:
    def __init__(self, min_support, max_len):
        if min_support is None and max_len is None:
            self.min_support = round(random.uniform(0.01, 0.50), 4)
            self.max_len = random.randint(2,7)
            
        else:
            self.min_support = min_support
            self.max_len = max_len
        
        self.rules = None
        self.fitness_score = self.calcular_fitness()

    def calcular_fitness(self): 
        frequent_itemsets = apriori(
            DADOS,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
        )
        
        self.rules = self.filter_rules(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4))
        mean_lift = self.rules['lift'].mean()
        mean_support = self.rules['support'].mean()

        #print(f"Fitness calculado: lift médio={mean_lift}, support médio={mean_support}")
        return mean_lift * mean_support 
    
    def filter_rules(self, temporary_rules):
        new_rules = temporary_rules[
            temporary_rules['antecedents'].apply(lambda items: any(str(i).startswith("ArrhythmiaClass=") for i in items)) |
            temporary_rules['consequents'].apply(lambda items: any(str(i).startswith("ArrhythmiaClass=") for i in items))
        ].copy()

        return new_rules.copy() 
    
    def __str__(self):
        return f"Individuo(min_support={self.min_support}, max_len={self.max_len}, media do lift={self.rules['lift'].mean()} media do support={self.rules['support'].mean()} fitness={self.fitness_score})"


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

        media_acuracia_geracao = [] 
        somat = 0
        for i in self.populacao_atual:
            somat += i.fitness_score    

        media_acuracia_geracao.append(somat/len(self.populacao_atual))

        for _ in tqdm(range(self.geracao-1)):
            self.crossover()
            self.mutar()
            self.acharMelhorIndividuo()

            soma = 0
            
            for i in self.populacao_atual:
                i.calcular_fitness()
                soma += i.fitness_score    

            media_acuracia_geracao.append(soma/len(self.populacao_atual))
        
        self.guardar_resultado()

        for i in range(len(media_acuracia_geracao)):
            print(f"Geração {i+1}: Média do fitness = {media_acuracia_geracao[i]}")

    def gerarPopulacao(self) -> list:
        populacao = []
        for i in range(self.individuo):
            populacao.append(Individuo(None, None))
        
        return populacao
    
    def acharMelhorIndividuo(self) -> Individuo:
        melhor = self.populacao_atual[0]
        for individuo in self.populacao_atual:
            if individuo.fitness_score > melhor.fitness_score:
                melhor = individuo
        return melhor

    def crossover(self):
        nova_geracao = []

        for _ in range(self.individuo//2):
        
            x = self.selecao()
            y = self.selecao()

            primeiro = Individuo(
                min_support=x.min_support,
                max_len=y.max_len
            )
            segundo = Individuo(
                min_support=y.min_support,
                max_len=x.max_len
            )
            nova_geracao.append(primeiro)
            nova_geracao.append(segundo)    
        
        self.populacao_atual = nova_geracao

    def mutar(self):
        for individuo in self.populacao_atual:
            if random.randint(1,100) <= self.mutacao:
                individuo.min_support = round(random.uniform(0.01, 0.50), 4)
                individuo.max_len = random.randint(2,6)
                individuo.fitness_score = individuo.calcular_fitness()

    def selecao(self):
        x, y= random.sample(range(0, self.individuo-1), 2)

        if self.populacao_atual[x].fitness_score > self.populacao_atual[y].fitness_score:
            return self.populacao_atual[x]
        else:
            return self.populacao_atual[y]
        
    def guardar_resultado(self):
        caminho_arquivo = f"resultado_GA.csv"
        print("✔ Melhor indivíduo encontrado: ", self.melhor_individuo)
        try:
            self.melhor_individuo.rules.to_csv(caminho_arquivo, index=False, encoding='utf-8')
            print(f"✔ Arquivo salvo com sucesso em: {caminho_arquivo}")
        except Exception as e:
            print("❌ Erro ao salvar o arquivo:", e)


import time
t1 = time.perf_counter()

ga = GA(individuo=10, geracao=10, mutacao=20)

t2 = time.perf_counter()
print(f"Tempo de execução: {(t2-t1):.2f}s")