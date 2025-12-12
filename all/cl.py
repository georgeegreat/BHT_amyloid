import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import matplotlib
import numpy as np  # Для np.where и np.array
import os  # Для создания директории, если ее нет

matplotlib.use("Agg")  # Убедитесь, что это используется для неинтерактивного режима

savedir = "graphics/"
# Создаем директорию для сохранения, если она не существует
if not os.path.exists(savedir):
    os.makedirs(savedir)
    print(f"Директория '{savedir}' создана.")

files = glob("*.csv")
plt.rcParams['font.size'] = 12

for file in files:
    print(f"Обработка файла: {file}")
    frame = pd.read_csv(file)
    cols = frame.columns

    # Извлекаем столбцы, заканчивающиеся на "_score"
    needle_cols = [i for i in cols if i.endswith("_score")]

    # Расчет bsum
    # Убедимся, что bsum рассчитывается корректно для всех столбцов needle_cols
    # Если needle_cols пуст, это вызовет ошибку, поэтому проверяем это
    if not needle_cols:
        print(f"В файле '{file}' не найдено столбцов, заканчивающихся на '_score'. Пропускаем.")
        continue

    # Инициализируем bsum с первым столбцом
    bsum_values = frame[needle_cols[0]].values.copy()

    # Суммируем остальные столбцы
    for c_name in needle_cols[1:]:
        bsum_values += frame[c_name].values

    # Преобразуем bsum_values в 0 или 1
    # Это будет массив/список, где 1, если сумма > 4, иначе 0
    bsum_indicator = [1 if i > 4 else 0 for i in bsum_values]

    plt.figure(figsize=(50, 22))

    # coef = 1 # Не используется в этом блоке, но оставлено для контекста

    legs = []  # Легенды для каждого графика

    # Итерируем по каждому столбцу, который нужно отобразить
    for col_name in needle_cols:
        legs.append(col_name)

        # Получаем значения для текущего столбца
        current_y_values = frame[col_name].values

        # Итерируем по сегментам линии
        for i in range(1, len(current_y_values)):
            # Определяем цвет сегмента на основе bsum_indicator
            segment_color = 'red' if bsum_indicator[i] == 1 or bsum_indicator[i - 1] == 1 else 'blue'
            # Если bsum 1 для текущей или предыдущей точки, красим сегмент в красный.
            # Это позволяет выделить сегмент, если он "входит" в зону 1.

            # Строим сегмент линии
            plt.plot(
                [i - 1, i],  # X-координаты текущего сегмента
                [current_y_values[i - 1], current_y_values[i]],  # Y-координаты текущего сегмента
                color=segment_color,
                linewidth=2 if segment_color == 'red' else 1
            )

    # Добавляем легенду для всех линий (она будет отображать только последний цвет, если не указать явно)
    # Для более точной легенды нужно будет делать фиктивные plot-ы.
    # Простой способ:
    plt.plot([], [], color='blue', label='bsum = 0')
    plt.plot([], [], color='red', label='bsum = 1')
    plt.legend(legs + ['bsum = 0', 'bsum = 1'])  # Добавляем легенду для цветов bsum

    # Или более сложный, но правильный способ для легенды:
    # (Раскомментируйте, если нужны только легенды для самих столбцов)
    # plt.legend(legs)

    tmp = file.split(".")[0]
    output_filename = f"{savedir}{tmp}_colored_by_bsum.jpg"
    plt.savefig(output_filename)
    plt.close()
    print(f"График для '{file}' сохранен как '{output_filename}'")

print("Все графики сгенерированы.")
