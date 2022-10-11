from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
import Model
import config


bot = Bot(token=config.TOKEN, parse_mode='HTML')
dp = Dispatcher(bot, storage=MemoryStorage())
clf = Model.Model()


async def start_bot(dp):
    try:
        print('start load')
        clf.load()
        print('stop load')
    except:
        await bot.send_message(str(config.admin_id), 'Ошибка загрузки модели')
    await bot.send_message(str(config.admin_id), 'Бот запущен')


async def stop_bot(dp):
    await bot.send_message(str(config.admin_id), 'Бот выключен')


@dp.message_handler(commands=['make_model'])
async def make_model(message: types.Message):
    clf.make_model()


@dp.message_handler(commands=['dump'])
async def dump(message: types.Message):
    clf.dump()


@dp.message_handler(commands=['load'])
async def load(message: types.Message):
    clf.load()


@dp.message_handler(content_types=['text'])
async def handler(message: types.Message):
    result = clf.pred_prob_agress(message.text)
    await message.reply(f'Недоброжелательность {result}')

if __name__ == '__main__':
    executor.start_polling(dp, on_startup=start_bot, on_shutdown=stop_bot, skip_updates=True)