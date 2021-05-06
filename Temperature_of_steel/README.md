## Прогнозирование потребления электроэнергии на этапе обработки стали металлургического предприятия

Необходимо проанализировать данные, обработать данные, извлечь значимые признаки и построить модель, которая предскажет температуру стали, что позволит уменьшить потребление электроэнергии на этапе обработки стали.
Метрикой качества является `MAE`. 

Целевое значение `МАЕ` на тестовых данных **<6.0**

### Обработка данных и EDA
Данные были исследованы, очищены от выбросов и объединены в один датафрейм.
![EDA](https://github.com/brut0/yandex.praktikum_ds_projects/blob/main/Temperature_of_steel/pics/EDA_temperature.jpg)

### Feature selection and feature engineering
Извлечены значимые признаки, отобраны существующие признаки и удалены лишние признаки по причине выявленной мультиколлинеарности.

![EDA](https://github.com/brut0/yandex.praktikum_ds_projects/blob/main/Temperature_of_steel/pics/feature_selection.jpg)

### Обучение и сравнение моделей
Были обучены различные модели регрессии и сравнены по метрике `МАЕ` кросс-валидацией.
![EDA](https://github.com/brut0/yandex.praktikum_ds_projects/blob/main/Temperature_of_steel/pics/model_comparison.jpg)

### Feature importance
Выявлены признаки не влияющие на целевой признак.
![EDA](https://github.com/brut0/yandex.praktikum_ds_projects/blob/main/Temperature_of_steel/pics/feature_importance.jpg)

### Используемые библиотеки:
- `numpy`
- `pandas`
- `sklearn`
- `catboost`
- `matplotlib`
- `seaborn`
- `shap`