# Полное руководство по настройке GitHub и Obsidian

## 1. Создание репозитория на GitHub
1. github.com → New repository
2. Назвать репозиторий (например, Telemost)
3. Сделать Public
4. НЕ ставить галочку "Initialize with README"

## 2. Настройка Personal Access Token
1. GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Выбрать все разрешения для repo
4. Сохранить токен - он понадобится вместо пароля при push

## 3. Создание локальной папки 
```bash
mkdir Telemost
cd Telemost
```

## 4. Три основных способа работы с файлами

### Способ 1: Отправка одного файла
```bash
cd путь/к/папке/Telemost
git init
git add 123.txt
git commit -m "123"
git branch -M main
git remote add origin https://github.com/ВАШ_АККАУНТ/Telemost.git
git push origin main
```

### Способ 2: Отправка всех файлов
```bash
cd путь/к/папке/Telemost
git add .
git commit -m "add all"
git push origin main
```

### Способ 3: Получение файлов с GitHub
```bash
cd путь/к/папке/Telemost
git pull origin main
```

## 5. Настройка Obsidian
1. Установить Obsidian
2. Создать vault в папке Telemost
3. Установить плагин Git:
   - Settings → Community plugins
   - Отключить Safe mode
   - Browse → найти "Git"
   - Установить и включить

## 6. Автоматическая синхронизация
В настройках плагина Git:
4. Settings → Git
5. Auto commit-and-sync interval: 5 (минут)
6. Auto push interval: 5
7. Auto pull interval: 5

## Заметки и решение проблем
- При первом push потребуется ввести логин GitHub и personal access token (не пароль!)
- После настройки автосинхронизации ручные команды git не требуются
- Если на GitHub и в локальной папке разные файлы:
  1. Сначала отправьте свои изменения (git push)
  2. Потом получите чужие (git pull)

## Проверка работы
8. Создайте тестовый файл в папке
9. Отправьте его через git push
10. Проверьте появление файла на GitHub
11. Попросите коллегу склонировать репозиторий

## Работа в команде
12. Коллега должен:
   - Склонировать репозиторий
   - Установить Obsidian
   - Настроить Git плагин
13. Дальше синхронизация будет работать автоматически