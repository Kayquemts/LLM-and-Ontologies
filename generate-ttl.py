"""
gerar_kgc_turtle.py
====================
Lê resultado_com_regras.csv, aplica mapeamento SNOMED CT e gera um
Knowledge Graph Completion (KGC) no formato RDF/Turtle (.ttl).

Saída: knowledge_graph.ttl

Estrutura do grafo
------------------
Entidades principais:
  :Patient_<id>          – paciente
  :Segment_<ecg_id>      – segmento ECG
  :Rule_<hash>           – regra de associação derivada pelo GA

Relações (além das do SNOMED):
  :hasSegment            – Patient → Segment
  :hasArrhythmiaClass    – Segment → ClinicalCondition (SNOMED)
  :hasActivity           – Segment → ActivityConcept   (SNOMED)
  :hasBodyPosition       – Segment → BodyPositionConcept
  :hasSleepWakeState     – Segment → SleepWakeConcept
  :hasGender             – Patient → GenderConcept
  :hasHeartRateCategory  – Segment → HeartRateConcept
  :hasMETCategory        – Segment → METConcept
  :hasAccelerationCategory – Segment → AccConcept
  :hasWeightCategory     – Patient → WeightConcept
  :hasAgeCategory        – Patient → AgeConcept
  :hasHeightCategory     – Patient → HeightConcept
  :derivedFromRule       – Segment → Rule  (KGC: link inferido)
  :ruleAntecedent        – Rule → Conceito SNOMED
  :ruleConsequent        – Rule → Conceito SNOMED
  :inferredCondition     – Segment → ClinicalCondition  (KGC: tripla completada)
"""

import re
import hashlib
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS

# ==============================================================================
# 1. NAMESPACES
# ==============================================================================

EX     = Namespace("http://example.org/arrhythmia#")
SCT    = Namespace("http://snomed.info/id/")
FHIR   = Namespace("http://hl7.org/fhir/")

# ==============================================================================
# 2. MAPEAMENTO SNOMED CT
# ==============================================================================
# Formato: termo_csv -> (sctid_ou_expr, label_legível, semantic_tag)
# Expressões compostas usam "+" (pós-coordenação)

SNOMED_MAP = {
    # -- Arritmias / condições clínicas --
    "AF (Atrial Fibrillation)":   ("49436004",           "Atrial fibrillation",          "disorder"),
    "NSR (Normal Sinus Rhythm)":  ("426285000",          "Normal sinus rhythm",           "finding"),
    "Noise":                       ("251143007",          "Noise artifact on ECG",         "finding"),
    "Others":                      ("10003008",           "Other cardiac arrhythmia",      "disorder"),

    # -- Atividade física --
    "activity_unknown":            ("261665006",          "Unknown activity",              "qualifier"),
    "activity_lying":              ("102538003",          "Recumbency",                    "finding"),
    "activity_sitting_standing":   ("33586001+10904000",  "Sitting and standing",          "finding"),
    "activity_cycling":            ("54921000087109",     "Cycling activity",              "finding"),
    "activity_slope_up":           ("282489003",          "Walking uphill",                "finding"),
    "activity_jogging":            ("1968006",            "Jogging",                       "finding"),
    "activity_slope_down":         ("282495002",          "Walking downhill",              "finding"),
    "activity_walking":            ("129006008",          "Walking",                       "finding"),
    "activity_sitting_lying":      ("33586001+102538003", "Sitting and lying",             "finding"),
    "activity_standing":           ("10904000",           "Standing",                      "finding"),
    "activity_sitting_lying_standing": ("33586001+102538003+10904000", "Sitting lying standing", "finding"),
    "activity_sitting":            ("33586001",           "Sitting",                       "finding"),
    "activity_not_worn":           ("262009000",          "Device not worn",               "qualifier"),

    # -- Posição corporal --
    "body_unknown":                ("261665006",          "Unknown body position",         "qualifier"),
    "body_lying_supine":           ("40199007",           "Supine body position",          "finding"),
    "body_lying_left":             ("272571008",          "Left lateral body position",    "finding"),
    "body_lying_prone":            ("1240000",            "Prone body position",           "finding"),
    "body_lying_right":            ("272570009",          "Right lateral body position",   "finding"),
    "body_upright":                ("249862003",          "Upright body position",         "finding"),
    "body_sitting_lying":          ("33586001+102538003", "Sitting and lying",             "finding"),
    "body_standing":               ("10904000",           "Standing body position",        "finding"),
    "body_not_worn":               ("262009000",          "Device not worn",               "qualifier"),

    # -- Sono / vigília --
    "wake":                        ("27625002",           "Awake",                         "finding"),
    "sleep":                       ("258158006",          "Sleep",                         "finding"),
    "not_worn":                    ("262009000",          "Device not worn",               "qualifier"),

    # -- Gênero --
    "gender_M":                    ("248153007",          "Male gender",                   "finding"),
    "gender_F":                    ("248152002",          "Female gender",                 "finding"),

    # -- Frequência cardíaca (bins) --
    "hr_bradycardia":              ("48867003",           "Bradycardia",                   "finding"),
    "hr_normal":                   ("301075000",          "Normal heart rate",             "finding"),
    "hr_elevated":                 ("3424008",            "Tachycardia",                   "finding"),
    "hr_tachycardia":              ("3424008",            "Tachycardia",                   "finding"),

    # -- Faixas etárias --
    "adulto_jovem":                ("133936004",          "Young adult",                   "finding"),
    "adulto":                      ("133936004",          "Adult",                         "finding"),
    "quase_idoso":                 ("133936004",          "Middle-aged adult",             "finding"),
    "idoso":                       ("105436006",          "Elderly",                       "finding"),

    # -- Peso (bins) --
    "peso_muito_baixo":            ("726527001+260362008","Very low body weight",          "finding"),
    "peso_baixo":                  ("726527001+62482003", "Low body weight",               "finding"),
    "peso_medio":                  ("726527001+1255665007","Normal body weight",           "finding"),
    "peso_alto":                   ("726527001+75540009", "High body weight",              "finding"),

    # -- Altura (bins) --
    "height_baixo":                ("1153637007+62482003","Low body height",               "finding"),
    "height_medio":                ("1153637007+1255665007","Normal body height",          "finding"),
    "height_alto":                 ("1153637007+75540009","High body height",              "finding"),
    "height_muito_alto":           ("1153637007+260360000","Very high body height",        "finding"),

    # -- MET (bins) --
    "met_1.0":                     ("698834005+62482003", "Low metabolic equivalent",      "finding"),
    "met_1.25":                    ("698834005+1255665007","Normal metabolic equivalent",  "finding"),
    "met_acima_1.25":              ("698834005+75540009", "High metabolic equivalent",     "finding"),

    # -- Aceleração (bins) --
    "acc_muito_baixa":             ("285659007+62482003", "Very low movement acceleration","observable"),
    "acc_baixa":                   ("285659007+62482003", "Low movement acceleration",     "observable"),
    "acc_moderada":                ("285659007+1255665007","Moderate movement acceleration","observable"),
    "acc_alta":                    ("285659007+75540009", "High movement acceleration",    "observable"),
}

# Propriedade RDF correspondente a cada coluna de contexto
PROP_MAP = {
    "heart_rate_bin":          EX.hasHeartRateCategory,
    "ActivityClass_mapped":    EX.hasActivity,
    "BodyPosition_mapped":     EX.hasBodyPosition,
    "NonWearSleepWake_mapped":  EX.hasSleepWakeState,
    "gender_mapped":           EX.hasGender,
    "MET_bin":                 EX.hasMETCategory,
    "acc_bin":                 EX.hasAccelerationCategory,
    "weight_bin":              EX.hasWeightCategory,
    "age_bin":                 EX.hasAgeCategory,
    "height_bin":              EX.hasHeightCategory,
    "ArrhythmiaClass":         EX.hasArrhythmiaClass,
}

# ==============================================================================
# 3. UTILITÁRIOS
# ==============================================================================

def uri_safe(text: str) -> str:
    """Converte string em fragmento URI seguro."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(text))

def sctid_to_uri(expr: str) -> URIRef:
    """
    Converte expressão SNOMED (simples ou pós-coordenada) em URI.
    Expressões compostas (A+B) recebem URI de post-coordination no namespace EX.
    """
    if "+" in expr:
        safe = "postcoord_" + expr.replace("+", "_")
        return EX[safe]
    return SCT[expr]

def concept_uri(term: str) -> URIRef | None:
    """Retorna a URI do conceito SNOMED para um termo do CSV."""
    entry = SNOMED_MAP.get(term)
    if not entry:
        return None
    sctid, _, _ = entry
    return sctid_to_uri(sctid)

def rule_uri(rule_str: str) -> URIRef:
    """Cria URI única para uma regra com base no seu conteúdo (hash MD5)."""
    h = hashlib.md5(rule_str.encode()).hexdigest()[:10]
    return EX[f"Rule_{h}"]

def parse_rule(rule_str: str):
    """
    Faz parse da coluna regra_ga.
    Formato: 'col=val AND col=val => col=val AND col=val'
    Retorna (antecedentes: list[str], consequentes: list[str])
    """
    if pd.isna(rule_str) or not rule_str.strip():
        return [], []
    parts = rule_str.split("=>")
    if len(parts) != 2:
        return [], []
    ant_raw  = [t.strip() for t in parts[0].split("AND")]
    cons_raw = [t.strip() for t in parts[1].split("AND")]
    return ant_raw, cons_raw

def extract_value(item: str) -> str:
    """Extrai o valor de uma expressão 'col=valor'."""
    return item.split("=", 1)[-1].strip() if "=" in item else item.strip()

# ==============================================================================
# 4. CONSTRUÇÃO DO GRAFO
# ==============================================================================

class KGCBuilder:
    def __init__(self):
        self.g = Graph()
        self.g.bind("ex",   EX)
        self.g.bind("sct",  SCT)
        self.g.bind("fhir", FHIR)
        self.g.bind("skos", SKOS)
        self.g.bind("owl",  OWL)

        self._define_ontology_header()
        self._define_classes()
        self._define_properties()
        self._define_snomed_concepts()

        self.defined_rules: set = set()  # evita duplicatas de regras

    # ------------------------------------------------------------------
    # 4a. CABEÇALHO DA ONTOLOGIA
    # ------------------------------------------------------------------
    def _define_ontology_header(self):
        onto = EX[""]
        self.g.add((onto, RDF.type,         OWL.Ontology))
        self.g.add((onto, RDFS.label,       Literal("Arrhythmia Context Knowledge Graph", lang="en")))
        self.g.add((onto, RDFS.comment,     Literal(
            "KGC gerado a partir de regras de associação do algoritmo genético "
            "aplicadas ao dataset de contexto de arritmias. "
            "Conceitos rastreados ao SNOMED CT.", lang="pt")))
        self.g.add((onto, OWL.versionInfo,  Literal("1.0.0")))

    # ------------------------------------------------------------------
    # 4b. CLASSES OWL
    # ------------------------------------------------------------------
    def _define_classes(self):
        classes = {
            EX.Patient:              "Patient",
            EX.ECGSegment:           "ECG Segment",
            EX.AssociationRule:      "Association Rule (GA)",
            EX.ClinicalCondition:    "Clinical Condition",
            EX.ActivityConcept:      "Physical Activity Concept",
            EX.BodyPositionConcept:  "Body Position Concept",
            EX.SleepWakeConcept:     "Sleep/Wake State Concept",
            EX.GenderConcept:        "Gender Concept",
            EX.HeartRateConcept:     "Heart Rate Category",
            EX.METConcept:           "MET Category",
            EX.AccelerationConcept:  "Movement Acceleration Category",
            EX.WeightConcept:        "Weight Category",
            EX.AgeConcept:           "Age Category",
            EX.HeightConcept:        "Height Category",
        }
        for uri, label in classes.items():
            self.g.add((uri, RDF.type,   OWL.Class))
            self.g.add((uri, RDFS.label, Literal(label, lang="en")))

    # ------------------------------------------------------------------
    # 4c. PROPRIEDADES OWL
    # ------------------------------------------------------------------
    def _define_properties(self):
        obj_props = {
            EX.hasSegment:             ("Patient",         "ECGSegment",            "has ECG segment"),
            EX.hasArrhythmiaClass:     ("ECGSegment",      "ClinicalCondition",     "has arrhythmia class"),
            EX.hasActivity:            ("ECGSegment",      "ActivityConcept",       "has physical activity"),
            EX.hasBodyPosition:        ("ECGSegment",      "BodyPositionConcept",   "has body position"),
            EX.hasSleepWakeState:      ("ECGSegment",      "SleepWakeConcept",      "has sleep/wake state"),
            EX.hasGender:              ("Patient",         "GenderConcept",         "has gender"),
            EX.hasHeartRateCategory:   ("ECGSegment",      "HeartRateConcept",      "has heart rate category"),
            EX.hasMETCategory:         ("ECGSegment",      "METConcept",            "has MET category"),
            EX.hasAccelerationCategory:("ECGSegment",      "AccelerationConcept",   "has acceleration category"),
            EX.hasWeightCategory:      ("Patient",         "WeightConcept",         "has weight category"),
            EX.hasAgeCategory:         ("Patient",         "AgeConcept",            "has age category"),
            EX.hasHeightCategory:      ("Patient",         "HeightConcept",         "has height category"),
            EX.ruleAntecedent:         ("AssociationRule", None,                    "rule antecedent concept"),
            EX.ruleConsequent:         ("AssociationRule", None,                    "rule consequent concept"),
            EX.derivedFromRule:        ("ECGSegment",      "AssociationRule",       "derived from association rule"),
            EX.inferredCondition:      ("ECGSegment",      "ClinicalCondition",     "KGC inferred condition"),
        }
        for uri, (domain, rng, label) in obj_props.items():
            self.g.add((uri, RDF.type,    OWL.ObjectProperty))
            self.g.add((uri, RDFS.label,  Literal(label, lang="en")))
            self.g.add((uri, RDFS.domain, EX[domain]))
            if rng:
                self.g.add((uri, RDFS.range, EX[rng]))

        data_props = {
            EX.patientID:    ("Patient",    XSD.string,  "patient identifier"),
            EX.segmentID:    ("ECGSegment", XSD.string,  "ECG segment identifier"),
            EX.rrInterval:   ("ECGSegment", XSD.string,  "RR interval"),
            EX.heartRate:    ("ECGSegment", XSD.float_,  "heart rate in bpm"),
            EX.timestamp:    ("ECGSegment", XSD.date,    "timestamp"),
            EX.ruleSupport:  ("AssociationRule", XSD.float_, "support"),
            EX.ruleConfidence:("AssociationRule",XSD.float_, "confidence"),
            EX.ruleLift:     ("AssociationRule", XSD.float_, "lift"),
            EX.ruleText:     ("AssociationRule", XSD.string, "rule in human-readable form"),
            EX.kgcStatus:    ("ECGSegment", XSD.string,  "KGC completion status"),
        }
        for uri, (domain, dtype, label) in data_props.items():
            self.g.add((uri, RDF.type,    OWL.DatatypeProperty))
            self.g.add((uri, RDFS.label,  Literal(label, lang="en")))
            self.g.add((uri, RDFS.domain, EX[domain]))
            self.g.add((uri, RDFS.range,  dtype))

    # ------------------------------------------------------------------
    # 4d. INSTÂNCIAS DOS CONCEITOS SNOMED
    # ------------------------------------------------------------------
    def _define_snomed_concepts(self):
        # Mapa de sctid -> classe OWL correspondente
        class_by_role = {
            "disorder":    EX.ClinicalCondition,
            "finding":     EX.ClinicalCondition,
            "qualifier":   EX.ClinicalCondition,
            "observable":  EX.AccelerationConcept,
        }
        # Mapeamento por prefixo do termo
        prefix_class = {
            "activity_": EX.ActivityConcept,
            "body_":     EX.BodyPositionConcept,
            "wake":      EX.SleepWakeConcept,
            "sleep":     EX.SleepWakeConcept,
            "not_worn":  EX.SleepWakeConcept,
            "gender_":   EX.GenderConcept,
            "hr_":       EX.HeartRateConcept,
            "met_":      EX.METConcept,
            "acc_":      EX.AccelerationConcept,
            "peso_":     EX.WeightConcept,
            "adulto":    EX.AgeConcept,
            "idoso":     EX.AgeConcept,
            "quase_":    EX.AgeConcept,
            "height_":   EX.HeightConcept,
        }

        for term, (sctid, label, tag) in SNOMED_MAP.items():
            uri = sctid_to_uri(sctid)

            # Determina a classe OWL
            owl_class = EX.ClinicalCondition  # fallback
            for prefix, cls in prefix_class.items():
                if term.startswith(prefix) or term == prefix.rstrip("_"):
                    owl_class = cls
                    break

            self.g.add((uri, RDF.type,   owl_class))
            self.g.add((uri, RDFS.label, Literal(label, lang="en")))
            self.g.add((uri, SKOS.notation, Literal(sctid)))

            # Pós-coordenação: relaciona partes
            if "+" in sctid:
                for part in sctid.split("+"):
                    self.g.add((uri, OWL.intersectionOf, SCT[part]))

            # Anotação SNOMED
            self.g.add((uri, EX.snomedExpression, Literal(sctid)))

    # ------------------------------------------------------------------
    # 5. ADICIONAR REGRA GA
    # ------------------------------------------------------------------
    def add_rule(self, rule_str: str, row: pd.Series) -> URIRef | None:
        """Registra uma regra de associação no grafo. Retorna a URI da regra."""
        if pd.isna(rule_str) or not rule_str.strip():
            return None

        r_uri = rule_uri(rule_str)
        if r_uri in self.defined_rules:
            return r_uri

        ant_raw, cons_raw = parse_rule(rule_str)

        self.g.add((r_uri, RDF.type,        EX.AssociationRule))
        self.g.add((r_uri, EX.ruleText,     Literal(rule_str)))

        # Métricas — presentes apenas no resultado_GA.csv; no resultado_com_regras
        # elas não estão disponíveis por linha, então omitimos com segurança.

        for item in ant_raw:
            val = extract_value(item)
            c_uri = concept_uri(val)
            if c_uri:
                self.g.add((r_uri, EX.ruleAntecedent, c_uri))

        for item in cons_raw:
            val = extract_value(item)
            c_uri = concept_uri(val)
            if c_uri:
                self.g.add((r_uri, EX.ruleConsequent, c_uri))

        self.defined_rules.add(r_uri)
        return r_uri

    # ------------------------------------------------------------------
    # 6. ADICIONAR PACIENTE
    # ------------------------------------------------------------------
    def add_patient(self, patient_id: str, row: pd.Series) -> URIRef:
        p_uri = EX[f"Patient_{uri_safe(patient_id)}"]
        self.g.add((p_uri, RDF.type,      EX.Patient))
        self.g.add((p_uri, EX.patientID,  Literal(str(patient_id), datatype=XSD.string)))

        # Atributos de nível paciente
        for col in ("gender_mapped", "weight_bin", "age_bin", "height_bin"):
            val = row.get(col)
            if pd.notna(val):
                c_uri = concept_uri(str(val))
                prop  = PROP_MAP.get(col)
                if c_uri and prop:
                    self.g.add((p_uri, prop, c_uri))

        return p_uri

    # ------------------------------------------------------------------
    # 7. ADICIONAR SEGMENTO ECG (KGC incluso)
    # ------------------------------------------------------------------
    def add_segment(self, row: pd.Series, patient_uri: URIRef) -> URIRef:
        ecg_id = str(row.get("ecg_id", "unknown"))
        s_uri  = EX[f"Segment_{uri_safe(ecg_id)}"]

        self.g.add((s_uri, RDF.type,      EX.ECGSegment))
        self.g.add((s_uri, EX.segmentID,  Literal(ecg_id, datatype=XSD.string)))
        self.g.add((patient_uri, EX.hasSegment, s_uri))

        rr = row.get("rr_interval")
        if pd.notna(rr):
            self.g.add((s_uri, EX.rrInterval, Literal(str(rr), datatype=XSD.string)))

        ts = row.get("timestamp")
        if pd.notna(ts):
            self.g.add((s_uri, EX.timestamp, Literal(str(ts)[:10], datatype=XSD.date)))

        hr = row.get("heart_rate")
        if pd.notna(hr):
            try:
                self.g.add((s_uri, EX.heartRate, Literal(float(hr), datatype=XSD.float_)))
            except (ValueError, TypeError):
                pass

        # Propriedades de contexto do segmento
        seg_cols = ("heart_rate_bin", "ActivityClass_mapped", "BodyPosition_mapped",
                    "NonWearSleepWake_mapped", "MET_bin", "acc_bin", "ArrhythmiaClass")
        for col in seg_cols:
            val = row.get(col)
            if pd.notna(val):
                c_uri = concept_uri(str(val))
                prop  = PROP_MAP.get(col)
                if c_uri and prop:
                    self.g.add((s_uri, prop, c_uri))

        # -----------------------------------------------------------------
        # KGC: aplica regra e infere triplas de completude
        # -----------------------------------------------------------------
        rule_str = row.get("regra_ga")
        r_uri    = self.add_rule(rule_str, row)

        if r_uri:
            # Segmento satisfaz os antecedentes → link com a regra
            self.g.add((s_uri, EX.derivedFromRule, r_uri))

            # Infere condições clínicas a partir dos consequentes da regra
            _, cons_raw = parse_rule(str(rule_str))
            inferred_any = False
            for item in cons_raw:
                val   = extract_value(item)
                c_uri = concept_uri(val)
                if c_uri:
                    self.g.add((s_uri, EX.inferredCondition, c_uri))
                    inferred_any = True

            status = "KGC_INFERRED" if inferred_any else "KGC_PARTIAL"
        else:
            status = "NO_RULE"

        self.g.add((s_uri, EX.kgcStatus, Literal(status, datatype=XSD.string)))
        return s_uri

    # ------------------------------------------------------------------
    # 8. SERIALIZAÇÃO
    # ------------------------------------------------------------------
    def serialize(self, path: str):
        self.g.serialize(destination=path, format="turtle")
        print(f"✔ Grafo serializado em: {path}")
        print(f"   → {len(self.g)} triplas RDF geradas")


# ==============================================================================
# 9. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    print("▶ Lendo resultado_com_regras.csv...")
    df = pd.read_csv("resultado_com_regras.csv", low_memory=False)
    print(f"   → {len(df)} linhas, {len(df.columns)} colunas")

    # Garante que colunas de contexto existam (podem não existir se o GA não as adicionou)
    context_cols = [
        "heart_rate_bin", "ActivityClass_mapped", "BodyPosition_mapped",
        "NonWearSleepWake_mapped", "gender_mapped", "MET_bin", "acc_bin",
        "weight_bin", "age_bin", "height_bin", "ArrhythmiaClass"
    ]
    for col in context_cols:
        if col not in df.columns:
            df[col] = None

    builder = KGCBuilder()

    patients_seen = {}
    stats = {"total": 0, "com_regra": 0, "sem_regra": 0, "erros": 0}

    print("▶ Construindo o Knowledge Graph...")
    for idx, row in df.iterrows():
        stats["total"] += 1
        try:
            pid = str(row.get("patient_id", row.get("patient", f"P{idx}")))

            if pid not in patients_seen:
                p_uri = builder.add_patient(pid, row)
                patients_seen[pid] = p_uri
            else:
                p_uri = patients_seen[pid]

            builder.add_segment(row, p_uri)

            if pd.notna(row.get("regra_ga")):
                stats["com_regra"] += 1
            else:
                stats["sem_regra"] += 1

        except Exception as e:
            stats["erros"] += 1
            print(f"  ⚠ Linha {idx}: {e}")

    print(f"\n📊 Estatísticas:")
    print(f"   Linhas processadas : {stats['total']}")
    print(f"   Com regra GA       : {stats['com_regra']}")
    print(f"   Sem regra (null)   : {stats['sem_regra']}")
    print(f"   Erros ignorados    : {stats['erros']}")
    print(f"   Pacientes únicos   : {len(patients_seen)}")
    print(f"   Regras únicas (GA) : {len(builder.defined_rules)}")

    builder.serialize("knowledge_graph.ttl")


if __name__ == "__main__":
    main()