import pandas as pd
import numpy as np
from sqlalchemy import create_engine, exc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import logging
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")


class CreditRiskModel:
    def __init__(self, db_user, db_password, db_host, db_name, db_port=3306):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        escaped_password = quote_plus(db_password)

        self.connection_string = (
            f"mysql+mysqlconnector://{db_user}:{escaped_password}@{db_host}:{db_port}/{db_name}"
            "?charset=utf8mb4&autocommit=true"
        )

        try:
            self.engine = create_engine(
                self.connection_string,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False,
            )
        except Exception as e:
            self.logger.error(f"Erro ao criar conexão com o banco: {e}")
            raise

        self.model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            criterion="gini",
        )
        self.columns = []
        self.feature_importance = None

    def load_and_prepare_data(self):
        try:
            query = "SELECT * FROM germancredit"
            self.logger.info("Carregando dados do banco...")

            df = pd.read_sql(query, con=self.engine)
            self.logger.info(f"Dados carregados: {len(df)} registros")

            if "id" in df.columns:
                df = df.drop(columns=["id"])

            df["pers"] = 3 - df["pers"]
            df["gastarb"] = 3 - df["gastarb"]

            df = self.create_engineered_features(df)

            self.columns = df.drop(columns=["kredit"]).columns.tolist()
            X = df.drop(columns=["kredit"])
            y = df["kredit"]

            self.logger.info("Dados preparados para treinamento.")
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        except exc.SQLAlchemyError as e:
            self.logger.error(f"Erro ao executar query SQL: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            raise

    def create_engineered_features(self, df):
        df = df.copy()

        df["debt_to_income_ratio"] = df["hoehe"] / (df["hoehe"] / df["laufzeit"])

        df["financial_stability"] = (
            (df["laufkont"] >= 3).astype(int)
            + (df["sparkont"] >= 3).astype(int)
            + (df["beszeit"] >= 3).astype(int)
            + (df["wohnzeit"] >= 3).astype(int)
            + (df["verm"] >= 3).astype(int)
        )

        df["risk_score"] = (
            (df["moral"] <= 1).astype(int) * 3
            + (df["rate"] <= 2).astype(int) * 2
            + (df["bishkred"] >= 3).astype(int) * 2
            + (df["beszeit"] <= 2).astype(int) * 1
            + (df["alter"] <= 25).astype(int) * 1
        )

        df["age_group"] = pd.cut(
            df["alter"], bins=[0, 25, 35, 50, 100], labels=[1, 2, 3, 4]
        ).astype(int)

        df["credit_amount_group"] = pd.cut(
            df["hoehe"], bins=[0, 2000, 5000, 10000, float("inf")], labels=[1, 2, 3, 4]
        ).astype(int)

        df["conservative_profile"] = (
            (df["verw"].isin([3, 4, 5])).astype(int)
            + (df["laufzeit"] <= 24).astype(int)
            + (df["hoehe"] <= 5000).astype(int)
        )

        return df

    def train(self, X_train, y_train):
        self.logger.info("Iniciando treinamento do modelo Decision Tree...")

        self.model.fit(X_train, y_train)

        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {"feature": self.columns, "importance": self.model.feature_importances_}
            ).sort_values("importance", ascending=False)

        self.logger.info("Modelo Decision Tree treinado com sucesso!")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("================ Avaliação do Modelo ===============")
        print(f"Acurácia: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        tn, fp, fn, tp = cm.ravel()
        precision_bad = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_bad = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_good = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_good = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nMétricas para Clientes RUINS (classe 0):")
        print(
            f"Precisão: {precision_bad:.3f} (dos preditos como ruins, quantos eram realmente ruins)"
        )
        print(
            f"Recall: {recall_bad:.3f} (dos realmente ruins, quantos foram detectados)"
        )

        print(f"\nMétricas para Clientes BONS (classe 1):")
        print(
            f"Precisão: {precision_good:.3f} (dos preditos como bons, quantos eram realmente bons)"
        )
        print(
            f"Recall: {recall_good:.3f} (dos realmente bons, quantos foram detectados)"
        )

        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=["Ruim", "Bom"]))

        if self.feature_importance is not None:
            print("\nTop 10 Features Mais Importantes:")
            print(self.feature_importance.head(10).to_string(index=False))

        return accuracy

    def close_connection(self):
        if hasattr(self, "engine"):
            self.engine.dispose()
            self.logger.info("Conexão com o banco fechada.")

    def predict_new_data(self, new_data):
        if len(new_data) != len(self.columns):
            raise ValueError(f"O dado inserido deve ter {len(self.columns)} atributos.")

        input_df = pd.DataFrame([new_data], columns=self.columns)

        prediction = self.model.predict(input_df)
        probability = self.model.predict_proba(input_df)[0]

        confidence_threshold = 0.6

        if prediction[0] == 1:
            if probability[1] >= confidence_threshold:
                return "BOM"
            else:
                return "MODERADO"
        else:
            if probability[0] >= confidence_threshold:
                return "RUIM"
            else:
                return "MODERADO"

    def get_prediction_details(self, new_data):
        if len(new_data) != len(self.columns):
            raise ValueError(f"O dado inserido deve ter {len(self.columns)} atributos.")

        input_df = pd.DataFrame([new_data], columns=self.columns)

        prediction = self.model.predict(input_df)
        probability = self.model.predict_proba(input_df)[0]

        risk_factors = self.analyze_risk_factors(new_data)

        return {
            "prediction": "BOM" if prediction[0] == 1 else "RUIM",
            "probability_good": probability[1],
            "probability_bad": probability[0],
            "confidence": max(probability),
            "risk_factors": risk_factors,
        }

    def analyze_risk_factors(self, new_data):
        risk_factors = []

        feature_map = {col: idx for idx, col in enumerate(self.columns)}

        if "laufkont" in feature_map and new_data[feature_map["laufkont"]] <= 2:
            risk_factors.append("Conta corrente com saldo negativo ou inexistente")

        if "moral" in feature_map and new_data[feature_map["moral"]] <= 1:
            risk_factors.append("Histórico de crédito problemático")

        if "laufzeit" in feature_map and new_data[feature_map["laufzeit"]] > 36:
            risk_factors.append("Prazo de pagamento muito longo")

        if "rate" in feature_map and new_data[feature_map["rate"]] <= 2:
            risk_factors.append("Alta taxa de comprometimento da renda")

        if "beszeit" in feature_map and new_data[feature_map["beszeit"]] <= 2:
            risk_factors.append("Tempo de emprego insuficiente")

        if "alter" in feature_map and new_data[feature_map["alter"]] < 25:
            risk_factors.append("Idade muito jovem (maior risco)")

        if "hoehe" in feature_map and new_data[feature_map["hoehe"]] > 15000:
            risk_factors.append("Valor do crédito muito alto")

        return risk_factors

    def show_input_order(self):
        print("\n========== Ordem dos dados de entrada ==========")
        for index, col in enumerate(self.columns):
            print(f"{index + 1}. {col}")


def main():
    load_dotenv()

    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "root")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "32768"))
    db_name = os.getenv("DB_NAME", "statlog")

    model = None
    try:
        model = CreditRiskModel(db_user, db_password, db_host, db_name, db_port)

        X_train, X_test, y_train, y_test = model.load_and_prepare_data()

        model.train(X_train, y_train)

        model.evaluate(X_test, y_test)

    except Exception as e:
        print(f"Erro na execução: {e}")
    finally:
        if model:
            model.close_connection()


if __name__ == "__main__":
    main()
