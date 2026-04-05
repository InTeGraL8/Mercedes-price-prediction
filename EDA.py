import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EDA:
    def __init__(self, train_df, target):
        self.train_df = train_df
        self.target = target
        self.num_col = [col for col in self.train_df.drop(self.target, axis=1).columns if self.train_df[col].dtype != 'O']


    def label(self):
        print("Информация о датасете:")
        print(f"Всего {len(self.train_df)} строк")
        print("Таблица с основной информацией")
        info_df = pd.DataFrame()
        info_df['Колонки'] = self.train_df.columns
        info_df['Пропуски'] = self.train_df.isna().sum().values
        info_df['Уникальные значения'] = self.train_df.nunique().values
        info_df['Среднее/мода'] = [self.train_df[col].mean() if self.train_df[col].dtype != 'O' else self.train_df[col].mode().iloc[0] for col in self.train_df.columns]
        info_df['Медиана/частота'] = [self.train_df[col].median() if self.train_df[col].dtype != 'O' else self.train_df[col].value_counts().iloc[0] for col in self.train_df.columns]
        info_df['Типы'] = self.train_df.dtypes.values
        print(info_df)
        print("Первые 5 строк")
        print(self.train_df.head(5))


    def pyplot(self):
        # Получаю классы и их кол-во в таргете
        val_counts = self.train_df[self.target].value_counts()


        # Создаю холст
        plt.figure(figsize=(9,9))
        
        # Визуализация таргета, создаю колонки x=классы таргета y=их количество
        plt.title("Таргет")

        # Создание колонок
        plt.bar(val_counts.index, val_counts.values)
        
        # Заголовки
        plt.xlabel(self.target)
        plt.ylabel("Value counts")

        # Показываю график
        plt.show()


        #Гистограмма числовых колонок
        axes = 3
        rows = int(abs(len(self.num_col) / axes))

        fig, axes = plt.subplots(rows, axes, figsize=(5 * axes, 5 * rows))
        axes = axes.flatten()
        
        for i, col in enumerate(self.num_col):
            # Создаю гистограмму
            axes[i].hist(self.train_df[self.num_col[i]])

            # Заголовки
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel(self.num_col[i])

            # Показываю график
        plt.show()
        

    # Ящик с усами
    def boxplot(self):
        # Беру все числовые колонки кроме таргета
        

        # Делаю сетку(3 столбца и строки равные int(кол-ву числовых колонок / столбцы))
        axes = 3
        rows = int(abs(len(self.num_col) / axes))

        plt.title('Ящики с усами')

        fig, axes = plt.subplots(rows, axes, figsize=(5 * axes, 5 * rows))

        axes = axes.flatten()

        for i, col in enumerate(self.num_col):
            sns.boxplot(x=self.target, y=col, data=self.train_df, ax=axes[i])

        plt.show()


    def corr_coef(self):
        coef = self.train_df[self.num_col].corr()
        plt.figure(figsize=(16,9))

        sns.heatmap(coef, vmin=-1, vmax=1, center=0, annot=True, cmap="coolwarm", square=True)

        plt.show()


    def pred_vs_target(self, y_true, y_pred, title="Predictions vs Reality"):
        plt.figure(figsize=(10, 6))
        
        # Ошибка для цвета
        error = np.abs(y_true - y_pred)
        scatter = plt.scatter(y_true, y_pred, c=error, cmap='viridis', alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Линия идеала
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Идеал')
        
        plt.colorbar(scatter, label='Абсолютная ошибка')
        plt.xlabel('Реальная цена ($)')
        plt.ylabel('Предсказанная цена ($)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()