import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

# Загрузка данных
interactions_df = pd.read_csv("dataset/user_scrobbles.csv")
titles_df = pd.read_csv("dataset/artist_list.csv")

# Преобразовываем в словарь
titles_df.index = titles_df["artist_id"]
title_dict = titles_df["artist_name"].to_dict()

# Создание матрицы взаимодействий
rows, r_pos = np.unique(interactions_df.values[:, 0], return_inverse=True)
cols, c_pos = np.unique(interactions_df.values[:, 1], return_inverse=True)
interactions_sparse = sparse.csr_matrix((interactions_df.values[:, 2], (r_pos, c_pos)))

# Нормализация и вычисление матрицы похожести
Pui = normalize(interactions_sparse, norm='l2', axis=1)
sim = Pui.T * Pui


# Функция для получения ID исполнителя по его имени
def get_artist_id(artist_name):
    for artist_id, name in title_dict.items():
        if name.lower() == artist_name.lower():
            return artist_id
    return None


# Запрос имени исполнителя у пользователя
artist_name = input("Введите имя исполнителя: ")

# Получение ID исполнителя
artist_id = get_artist_id(artist_name)

if artist_id is not None:
    # Получение списка похожих исполнителей, исключая самого исполнителя
    similar_indices = sim[artist_id - 1].toarray().argsort()[0][-21:]
    similar_indices = [i for i in similar_indices if i != artist_id - 1]

    # Вывод похожих исполнителей
    out_pr = [title_dict[i + 1] for i in similar_indices]
    print("Похожие исполнители:")
    for i in out_pr:
        print(i)
else:
    print("Исполнитель не найден.")

