#!/usr/bin/env python3

FIELD_TRANSLATIONS = {
    "laufkont": "status_conta_corrente",
    "laufzeit": "duracao_credito",
    "moral": "historico_credito",
    "verw": "proposito",
    "hoehe": "valor_credito",
    "sparkont": "conta_poupanca",
    "beszeit": "duracao_emprego",
    "rate": "taxa_prestacao",
    "famges": "status_pessoal_sexo",
    "buerge": "outros_devedores",
    "wohnzeit": "residencia_atual",
    "verm": "propriedade",
    "alter": "idade",
    "weitkred": "outros_planos_prestacao",
    "wohn": "habitacao",
    "bishkred": "numero_creditos",
    "beruf": "trabalho",
    "pers": "pessoas_responsaveis",
    "telef": "telefone",
    "gastarb": "trabalhador_estrangeiro",
    "kredit": "risco_credito",
}

FIELD_DISPLAY_NAMES = {
    "laufkont": "Status da Conta Corrente",
    "laufzeit": "Duração do Crédito (meses)",
    "moral": "Histórico de Crédito",
    "verw": "Propósito do Crédito",
    "hoehe": "Valor do Crédito (DM)",
    "sparkont": "Conta Poupança",
    "beszeit": "Duração do Emprego",
    "rate": "Taxa de Prestação (%)",
    "famges": "Status Pessoal e Sexo",
    "buerge": "Outros Devedores/Garantidores",
    "wohnzeit": "Tempo na Residência Atual",
    "verm": "Propriedades",
    "alter": "Idade (anos)",
    "weitkred": "Outros Planos de Prestação",
    "wohn": "Tipo de Habitação",
    "bishkred": "Número de Créditos Existentes",
    "beruf": "Ocupação/Trabalho",
    "pers": "Pessoas Dependentes",
    "telef": "Possui Telefone",
    "gastarb": "Trabalhador Estrangeiro",
    "kredit": "Risco de Crédito",
}

FIELD_VALUES = {
    "laufkont": {
        1: "Sem conta corrente",
        2: "Saldo < 0 DM",
        3: "0 <= Saldo < 200 DM",
        4: "Saldo >= 200 DM / salário por pelo menos 1 ano",
    },
    "moral": {
        0: "Atraso no pagamento no passado",
        1: "Conta crítica/outros créditos em outro lugar",
        2: "Sem créditos anteriores/todos pagos corretamente",
        3: "Créditos existentes pagos corretamente até agora",
        4: "Todos os créditos neste banco pagos corretamente",
    },
    "verw": {
        0: "Outros",
        1: "Carro (novo)",
        2: "Carro (usado)",
        3: "Móveis/equipamentos",
        4: "Rádio/televisão",
        5: "Eletrodomésticos",
        6: "Reparos",
        7: "Educação",
        8: "Férias",
        9: "Retreinamento",
        10: "Negócios",
    },
    "sparkont": {
        1: "Desconhecido/sem conta poupança",
        2: "Saldo < 100 DM",
        3: "100 <= Saldo < 500 DM",
        4: "500 <= Saldo < 1000 DM",
        5: "Saldo >= 1000 DM",
    },
    "beszeit": {
        1: "Desempregado",
        2: "< 1 ano",
        3: "1 <= tempo < 4 anos",
        4: "4 <= tempo < 7 anos",
        5: ">= 7 anos",
    },
    "rate": {1: ">= 35%", 2: "25% <= taxa < 35%", 3: "20% <= taxa < 25%", 4: "< 20%"},
    "famges": {
        1: "Homem: divorciado/separado",
        2: "Mulher: não solteira ou Homem: solteiro",
        3: "Homem: casado/viúvo",
        4: "Mulher: solteira",
    },
    "buerge": {1: "Nenhum", 2: "Co-requerente", 3: "Garantidor"},
    "wohnzeit": {
        1: "< 1 ano",
        2: "1 <= tempo < 4 anos",
        3: "4 <= tempo < 7 anos",
        4: ">= 7 anos",
    },
    "verm": {
        1: "Desconhecido/sem propriedade",
        2: "Carro ou outros bens",
        3: "Acordo de poupança/seguro de vida",
        4: "Imóvel",
    },
    "weitkred": {1: "Banco", 2: "Lojas", 3: "Nenhum"},
    "wohn": {1: "Gratuito", 2: "Aluguel", 3: "Próprio"},
    "bishkred": {
        1: "1 crédito",
        2: "2-3 créditos",
        3: "4-5 créditos",
        4: ">= 6 créditos",
    },
    "beruf": {
        1: "Desempregado/não qualificado - não residente",
        2: "Não qualificado - residente",
        3: "Empregado qualificado/funcionário",
        4: "Gerente/autônomo/funcionário altamente qualificado",
    },
    "pers": {1: "3 ou mais pessoas", 2: "0 a 2 pessoas"},
    "telef": {1: "Não", 2: "Sim (no nome do cliente)"},
    "gastarb": {1: "Sim", 2: "Não"},
    "kredit": {0: "Ruim (Alto Risco)", 1: "Bom (Baixo Risco)"},
}

NUMERIC_FIELDS = ["laufzeit", "hoehe", "alter"]

FIELD_DESCRIPTIONS = {
    "laufkont": "Status da conta corrente do cliente, indicando o saldo atual e histórico bancário",
    "laufzeit": "Duração solicitada para o crédito em meses",
    "moral": "Histórico de pagamento de créditos anteriores do cliente",
    "verw": "Finalidade ou propósito para o qual o crédito será utilizado",
    "hoehe": "Valor total do crédito solicitado em marcos alemães (DM)",
    "sparkont": "Status da conta poupança do cliente e valor aproximado",
    "beszeit": "Tempo de trabalho no emprego atual",
    "rate": "Percentual da renda comprometida com a prestação",
    "famges": "Status civil e sexo do solicitante",
    "buerge": "Existência de co-devedores ou garantidores",
    "wohnzeit": "Tempo que o cliente reside no endereço atual",
    "verm": "Tipos de propriedades ou bens que o cliente possui",
    "alter": "Idade do solicitante em anos",
    "weitkred": "Existência de outros planos de financiamento ou prestações",
    "wohn": "Tipo de moradia do cliente (própria, alugada, etc.)",
    "bishkred": "Quantidade de créditos que o cliente já possui neste banco",
    "beruf": "Tipo de ocupação e nível de qualificação do cliente",
    "pers": "Número de pessoas que dependem financeiramente do cliente",
    "telef": "Se o cliente possui telefone registrado em seu nome",
    "gastarb": "Se o cliente é um trabalhador estrangeiro",
    "kredit": "Classificação do risco de crédito (variável alvo)",
}


def get_field_display_name(field_name):
    return FIELD_DISPLAY_NAMES.get(field_name, field_name)


def get_field_description(field_name):
    return FIELD_DESCRIPTIONS.get(field_name, "")


def get_value_description(field_name, value):
    if field_name in FIELD_VALUES:
        return FIELD_VALUES[field_name].get(value, str(value))
    return str(value)


def is_numeric_field(field_name):
    return field_name in NUMERIC_FIELDS


def get_field_options(field_name):
    if field_name in FIELD_VALUES:
        return FIELD_VALUES[field_name]
    return {}


def translate_field_name(german_name):
    return FIELD_TRANSLATIONS.get(german_name, german_name)


TEST_CASES = {
    "Cliente Ideal": {
        "description": "Cliente com perfil excelente - aprovação praticamente garantida",
        "values": {
            "laufkont": 4,
            "laufzeit": 12,
            "moral": 4,
            "verw": 3,
            "hoehe": 1500,
            "sparkont": 5,
            "beszeit": 5,
            "rate": 4,
            "famges": 3,
            "buerge": 1,
            "wohnzeit": 4,
            "verm": 4,
            "alter": 40,
            "weitkred": 3,
            "wohn": 3,
            "bishkred": 1,
            "beruf": 4,
            "pers": 2,
            "telef": 2,
            "gastarb": 2,
        },
    },
    "Cliente de Risco": {
        "description": "Cliente com múltiplos fatores de risco - negação recomendada",
        "values": {
            "laufkont": 1,
            "laufzeit": 48,
            "moral": 0,
            "verw": 8,
            "hoehe": 12000,
            "sparkont": 1,
            "beszeit": 1,
            "rate": 1,
            "famges": 1,
            "buerge": 1,
            "wohnzeit": 1,
            "verm": 1,
            "alter": 22,
            "weitkred": 1,
            "wohn": 2,
            "bishkred": 4,
            "beruf": 1,
            "pers": 1,
            "telef": 1,
            "gastarb": 1,
        },
    },
    "Cliente Médio": {
        "description": "Cliente com perfil misto - requer análise detalhada",
        "values": {
            "laufkont": 1,
            "laufzeit": 30,
            "moral": 2,
            "verw": 8,
            "hoehe": 6000,
            "sparkont": 1,
            "beszeit": 2,
            "rate": 1,
            "famges": 2,
            "buerge": 1,
            "wohnzeit": 1,
            "verm": 1,
            "alter": 25,
            "weitkred": 1,
            "wohn": 2,
            "bishkred": 3,
            "beruf": 2,
            "pers": 1,
            "telef": 1,
            "gastarb": 2,
        },
    },
}
