# Участник 3: модели, сравнение, разбор ошибок — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** обучить Random Forest и HistGradientBoosting на разбиении участника 2, подобрать порог по F1 только на обучающей части, сравнить с логистической регрессией с флагами аномалий и без них, разобрать ошибки и один раз оценить выбранную модель на test.

**Architecture:** вспомогательный код лежит в `participant3.py` рядом с `credit_pipeline.py` и покрыт тестами. Ноутбук `Participant3_Models_Comparison.ipynb` вызывает эти функции, объясняет каждый шаг и сохраняет результаты в `artifacts/participant3/`.

**Tech Stack:** Python, scikit-learn 1.9.1 (`RandomForestClassifier`, `HistGradientBoostingClassifier`, `TunedThresholdClassifierCV`), pandas, matplotlib, unittest. Новых зависимостей нет: XGBoost и LightGBM в `requirements.lock.txt` отсутствуют, а `HistGradientBoostingClassifier` относится к тому же семейству моделей.

**Spec:** задание `project-5/main_english.ipynb` в `Project_eng.zip`, раздел «Часть 3» в `PROGRESS.md`, раздел «For participant 3» в `README.md`.

## Почему выбраны эти методы (по результатам поиска)

- В статьях о кредитном дефолте ансамбли деревьев (Random Forest, градиентный бустинг) обычно обходят логистическую регрессию, но победитель зависит от данных. Поэтому сравниваем две модели, а не одну.
- Подбор порога часто даёт больше, чем ресэмплинг (SMOTE и т. п.). Порог подбираем через `TunedThresholdClassifierCV(scoring="f1")` на 5 фолдах обучающей части, без SMOTE.
- `HistGradientBoostingClassifier` сам работает с пропусками и поддерживает `class_weight="balanced"`.

Источники: [arXiv 1907.12996](https://export.arxiv.org/pdf/1907.12996), [SciTePress 2025](https://www.scitepress.org/publishedPapers/2025/137012/pdf/index.html), [MDPI Systems 13/7/581](https://www.mdpi.com/2079-8954/13/7/581), [sklearn: TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html), [sklearn: cost-sensitive learning](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html).

## Global Constraints

- Разбиение только через `cp.load_split()` (5 400 / 1 350 строк, `random_state=42`).
- `RANDOM_STATE = 42` у всех моделей и фолдов.
- Метрики для класса 1: Precision, Recall, F1.
- Строки test не удалять, предсказания нужны для всех 750 строк.
- `cp.load_test_labels()` вызывается в ноутбуке один раз, в последнем разделе, после выбора модели.
- Validation не участвует в подборе параметров и порога. На нём только сравниваем модели и выбираем финальную.
- Базовая модель для сравнения: логистическая регрессия, balanced, `build_preprocessor(scale=True)`, validation F1 0.536 (без флагов 0.472).
- Текст ноутбука на русском, без AI-штампов (прогнать через `/deslop`).

## Review Focus

1. Утечка порога: если порог подобран на validation, F1 завышен. Тест проверяет, что `tune_threshold` обучается только на переданных строках.
2. Сравнение «с флагами / без флагов» должно идти на одинаковом разбиении и пороге, подобранном отдельно для каждого варианта.
3. Пропуски в test (1 400 строк без Annual Income и Credit Score в train, такие же есть в test) не должны ронять `predict`. Тест прогоняет модель на `X_test`.
4. `y_pred` на test содержит ровно 750 значений из {0, 1}.
5. Деревья получают `scale=False`, логистическая регрессия получает `scale=True`.

---

### Task 1: Модуль `participant3.py`

**Files:**
- Create: `participant3.py`
- Test: `tests/test_participant3.py`

**Interfaces:**
- Consumes: `cp.build_preprocessor(use_anomaly_flags: bool, scale: bool)`, `cp.load_data()`, `cp.load_split()`, `cp.RANDOM_STATE`.
- Produces:
  - `make_models(use_anomaly_flags: bool = True) -> dict[str, Pipeline]` с ключами `"logreg"`, `"random_forest"`, `"hist_gb"`. Каждый Pipeline состоит из шагов `("pre", ...)` и `("model", ...)`.
    - logreg: `LogisticRegression(class_weight="balanced", max_iter=1000)` и препроцессор с `scale=True`.
    - random_forest: `RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42)` и `scale=False`.
    - hist_gb: `HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=42)` и `scale=False`.
  - `class1_metrics(y_true, y_pred) -> dict[str, float]` с ключами `precision`, `recall`, `f1` (`zero_division=0`).
  - `tune_threshold(pipe: Pipeline, X, y) -> TunedThresholdClassifierCV`: `scoring="f1"`, `cv=StratifiedKFold(5, shuffle=True, random_state=42)`, `store_cv_results=True`, обучен на (X, y).

- [ ] **Step 1: Написать тесты**

```python
class Participant3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.test = cp.load_data()
        cls.tr, _ = cp.load_split()

    def test_class1_metrics(self):
        m = p3.class1_metrics([0, 1, 1, 0], [0, 1, 0, 1])
        self.assertEqual(m, {"precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(p3.class1_metrics([0, 1], [0, 0])["f1"], 0.0)

    def test_models_and_scaling(self):
        for flags in (True, False):
            models = p3.make_models(flags)
            self.assertEqual(set(models), {"logreg", "random_forest", "hist_gb"})
        # деревья без масштабирования: money-трансформер = SimpleImputer
        money = p3.make_models()["hist_gb"]["pre"]["columns"].transformers[0][1]
        self.assertIsInstance(money, SimpleImputer)

    def test_tune_threshold_uses_only_given_rows_and_predicts_test(self):
        Xs, ys = self.X.loc[self.tr[:1500]], self.y.loc[self.tr[:1500]]
        tuned = p3.tune_threshold(p3.make_models()["hist_gb"], Xs, ys)
        self.assertTrue(0 < tuned.best_threshold_ < 1)
        self.assertEqual(len(tuned.cv_results_["thresholds"]), 100)
        pred = tuned.predict(self.test)
        self.assertEqual(len(pred), 750)
        self.assertLessEqual(set(pred), {0, 1})
```

- [ ] **Step 2:** `python -m unittest tests.test_participant3 -v` → FAIL (`No module named 'participant3'`).
- [ ] **Step 3:** реализовать три функции по сигнатурам выше.
- [ ] **Step 4:** `python -m unittest discover -s tests -v` → все тесты проходят, включая 5 старых.
- [ ] **Step 5:** commit `Add participant 3 model helpers`.

### Task 2: Ноутбук `Participant3_Models_Comparison.ipynb`

**Files:**
- Create: `Participant3_Models_Comparison.ipynb`
- Modify: `run_participant2.py`: имя ноутбука берётся из `sys.argv[1]`, по умолчанию остаётся ноутбук участника 2.
- Create (артефакты при запуске): `artifacts/participant3/cv_comparison.csv`, `validation_comparison.csv`, `validation_predictions.csv`, `test_metrics.json`, `figures/*.png`.

**Interfaces:**
- Consumes: Task 1, `artifacts/participant2/baseline_validation_predictions.csv`.

Разделы ноутбука (в каждом markdown-ячейка: что делаем и почему, после результата пара строк вывода):

- [ ] **Step 1. Введение:** цель, ссылки на части 1–2, базовая модель, которую нужно побить (F1 0.536). Коротко про RF, бустинг и подбор порога, со ссылками на источники выше.
- [ ] **Step 2. CV на обучающей части:** 3 модели × {с флагами, без флагов}. `cross_validate` с `scoring=["precision","recall","f1","roc_auc"]` при пороге 0.5, затем `tune_threshold` и его CV-оценка (`best_score_`, `best_threshold_`). Результат сохраняется в `cv_comparison.csv`.
- [ ] **Step 3. Validation:** 6 подобранных моделей, обученных на `idx_tr`, оцениваются на `idx_va`. Таблица Precision/Recall/F1 рядом со строкой baseline из `artifacts/participant2/baseline_validation_metrics.csv`, сохраняется в `validation_comparison.csv`. Графики: PR-кривые и матрицы ошибок.
- [ ] **Step 4. Важность признаков:** `permutation_importance` на validation для лучшей древесной модели, отдельно для вариантов с флагами и без. Это отвечает на открытый вопрос «сигнал или утечка»: смотрим, какую долю F1 дают два флага.
- [ ] **Step 5. Разбор ошибок:** FN и FP лучшей модели по `Term`, `Purpose`, `Home Ownership`, флагам и пропускам в Credit Score. Сравнение с ошибками baseline по `row`: какие строки исправлены, какие сломаны. Результат сохраняется в `validation_predictions.csv` (`row, y_true, proba, y_pred, model`).
- [ ] **Step 6. Выбор модели:** правило фиксируется до test: максимум validation F1, при разнице < 0.01 берём модель проще (logreg < RF < HGB). Финальная модель переобучается на `idx_tr ∪ idx_va` с порогом, заново подобранным через `tune_threshold` на этих строках.
- [ ] **Step 7. Test, единственный вызов:** `cp.load_test_labels()`, метрики класса 1, `assert len(pred) == 750`. Результат сохраняется в `test_metrics.json`.
- [ ] **Step 8. Вывод:** таблица всех моделей, ответы на вопросы про флаги и смысл класса 1, ограничения (EDA видел весь train, поэтому validation не полностью слепой).
- [ ] **Step 9:** `python run_participant2.py Participant3_Models_Comparison.ipynb` выполняется без ошибок, артефакты на месте.
- [ ] **Step 10:** прогнать markdown-ячейки через `/deslop` (file mode, код не трогать).
- [ ] **Step 11:** commit `Add participant 3 model comparison notebook`.

### Task 3: Документация

**Files:**
- Modify: `PROGRESS.md` (часть 3 → ✅, таблица метрик, история), `README.md` (статус части 3, команда запуска).

- [ ] **Step 1:** внести итоговые цифры из `validation_comparison.csv` и `test_metrics.json`.
- [ ] **Step 2:** `/deslop` для изменённых абзацев.
- [ ] **Step 3:** `python -m unittest discover -s tests -v` → PASS.
- [ ] **Step 4:** commit `Update progress for participant 3`.
