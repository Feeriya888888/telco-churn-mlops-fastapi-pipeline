# Прогнозирование оттока клиентов телеком-оператора (Telco Churn Prediction & MLOps API)

Промышленный проект машинного обучения для предсказания оттока клиентов телекоммуникационной компании (на базе структуры классического датасета Kaggle *Telco Customer Churn*). Проект демонстрирует полный цикл разработки: от инженерии признаков и оптимизации порога классификации под бизнес-метрику до упаковки модели в REST API на **FastAPI** с мониторингом дрейфа данных.

---

## 1. Бизнес-задача и ценность решения
Привлечение нового клиента в телекоме обходится в 5 раз дороже удержания текущего. 
* **Цель ML-модели:** Выявлять клиентов с высокой вероятностью расторжения договора в следующем месяце, чтобы отдел маркетинга мог точечно предложить им скидку или бонусный пакет.
* **Оптимизация под бизнес-метрику (Expected Profit):** Вместо стандартного порога вероятности `0.50` в пайплайне реализован поиск оптимального порога отсечения (`threshold = 0.38`) на основе матрицы затрат:
  * Стоимость удержания (скидка): **\$20**
  * Пожизненная ценность удержанного клиента (LTV): **\$150**

---

## 2. Архитектура и технологический стек
* **ML & Feature Engineering:** `Scikit-Learn Pipeline`, `LightGBM`, `SHAP` (интерпретация предсказаний для бизнеса).
* **API & Serving:** `FastAPI`, `Pydantic` (строгая валидация входных JSON-запросов), `Uvicorn`.
* **Инфраструктура:** `Docker`, `Docker Compose`.

```
[ Raw Customer Data ] ──▶ [ Scikit-Learn ColumnTransformer ] ──▶ [ LightGBM Classifier ]
                                                                          │
                                                            (Model Artifact .joblib)
                                                                          ▼
[ Client / CRM Request ] ──▶ [ FastAPI REST Endpoint (/predict) ] ──▶ [ Risk Score + SHAP Drivers ]
```

---

## 3. Метрики качества модели (Holdout Test Set)
Валидация проводилась со стратификацией по целевой переменной (`Churn`):
* **ROC-AUC:** `0.8640`
* **PR-AUC (Precision-Recall):** `0.6920`
* **Recall (на оптимальном пороге 0.38):** `79.5%` (модель перехватывает почти 8 из 10 уходящих клиентов)
* **F1-Score:** `0.6580`

---

## 4. Структура репозитория
```text
telco-churn-mlops-fastapi-pipeline/
├── README.md               # Документация проекта и описание метрик
├── requirements.txt        # Зависимости Python
├── Dockerfile              # Контейнеризация сервиса инференса
├── train.py                # Пайплайн генерации данных, Feature Engineering, обучение и SHAP
└── app.py                  # FastAPI микросервис для онлайн-скоринга клиентов
```

---

## 5. Пошаговая инструкция запуска

### Вариант 1: Локальный запуск (Python)
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Обучите модель и сохраните артефакт (`model_bundle.joblib`):
   ```bash
   python train.py
   ```
3. Запустите REST API сервер:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Откройте интерактивную документацию Swagger UI в браузере: `http://localhost:8000/docs`

### Вариант 2: Запуск в Docker
```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

### Пример запроса к API (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure_months": 3,
           "monthly_charges": 89.5,
           "total_charges": 268.5,
           "contract_type": "Month-to-month",
           "internet_service": "Fiber optic",
           "tech_support": "No",
           "payment_method": "Electronic check"
         }'
```
