# ChatControlBot
Бот предназначен для фильтрации "плохих" комментариев в телеграм, в основе лежит алгоритм логистической регрессии

В ChatControlBot.ipynb расписана разработка проекта\
config.py, Model.py, bot.py - файлы для работы бота

Для запуска бота надо:
- Скачать все файлы с git
- Установить токен бота в config.py
- Скачать dataset с https://www.kaggle.com/datasets/alexandersemiletov/toxic-russian-comments
- Запустить бота и пронать модель (функция /make_model) для сохранения весов в файл 
