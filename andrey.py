import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import traceback
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import pyrogram
import pyrogram.errors
from anthropic import AsyncAnthropic
from pyrogram import Client, filters, idle
from pyrogram.enums import ChatAction
from pyrogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from vosk import KaldiRecognizer, Model

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AIChatHistory:
    """Manages chat history storage and analysis"""

    user_id: int
    start_time: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, str]] = field(default_factory=list)

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.messages = []
        self.filename = (
            f"reports/{user_id}_history_{datetime.now().strftime('%H_%M_%d_%m_%y')}.md"
        )

    def add_message(self, role: str, content: str):
        """Add message to history, extracting only text content"""
        if role == "assistant":
            try:
                # Parse JSON response to get only the Reply content
                data = json.loads(content)
                content = data.get("Reply", "")
            except json.JSONDecodeError:
                pass

        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def save_to_file(self) -> str:
        """Save chat history to file with formatted timestamp"""
        with open(self.filename, "w", encoding="utf-8") as f:
            for msg in self.messages:
                f.write(
                    f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content']}\n"
                )
                f.write("-" * 80 + "\n")

        return self.filename

    @property
    def duration(self) -> str:
        """Get conversation duration in readable format"""
        duration = datetime.now() - self.start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        return f"{minutes}m {seconds}s"


@dataclass
class UserConversation:
    """Manages user conversation state and tracking"""

    user_id: int
    messages: list = field(default_factory=list)
    last_message_time: datetime = field(default_factory=datetime.now)
    conversation_active: bool = True
    timer_task: Optional[asyncio.Task] = None
    system_prompt: str = field(default="")
    is_initialized: bool = False
    message_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_response: Optional[str] = None
    is_processing: bool = False
    chat_history: Optional[AIChatHistory] = None
    last_message_id: Optional[int] = None
    start_message_id: Optional[int] = None
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_bot_message: Optional[Message] = None

    def __post_init__(self):
        """Initialize chat history after instance creation"""
        self.chat_history = AIChatHistory(self.user_id)
        logger.info(f"Chat history initialized for user {self.user_id}")

    async def cleanup(self) -> Optional[str]:
        """Clean method with enhanced logging"""
        logger.info(f"[CLEANUP-START] Beginning cleanup for user {self.user_id}")

        try:
            if self.timer_task and not self.timer_task.done():
                self.timer_task.cancel()
                try:
                    await self.timer_task
                except:
                    pass

            history_file = None
            if self.chat_history:
                history_file = self.chat_history.save_to_file()

            return history_file

        except Exception as e:
            logger.error(
                f"[CLEANUP-ERROR] Error in cleanup for {self.user_id}: {e}\n{traceback.format_exc()}"
            )
            return None
        finally:
            logger.info(f"[CLEANUP-END] Cleanup completed for {self.user_id}")


class DialogueAnalyzer:
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.claude_client = anthropic_client

    async def create_report(
        self, chat_history_path: str, user_id: str
    ) -> tuple[str, str] | None:
        """Asynchronous function to create analysis report"""
        logger.info(f"Starting report creation for user {user_id}")

        try:
            async with aiofiles.open(
                "prompts/estimated.md", "r", encoding="utf-8"
            ) as f:
                criteria = await f.read()

            async with aiofiles.open(chat_history_path, "r", encoding="utf-8") as f:
                dialogue = await f.read()

            prompt = f"""
You are an experienced Salesman evaluating a candidate based on their dialogue. Create a structured report in markdown format. Report should be in Russian language.

Required sections:

1. Evaluation (detailed):
   - Analysis based on these criteria:
{criteria}

2. Score:
   - Overall score (0-100)
   - Brief justification

3. Recommendation:
   - Clear hire/no hire decision
   - Key reasons

4. Strengths and Weaknesses:
   - Top 3 strengths
   - Areas for improvement

Here's the dialogue to analyze:
{dialogue}
"""
            response = await self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            report = response.content[0].text

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/{user_id}_report_{timestamp}.md"
            async with aiofiles.open(report_path, "w", encoding="utf-8") as f:
                await f.write(f"# Analysis Report for User {user_id}\n")
                await f.write(
                    f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                await f.write(report)

            return report, report_path

        except Exception as e:
            logger.error(
                f"Error in create_report: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            return None


class VoiceProcessor:
    def __init__(self, model_path: str = "./vosk-model-ru"):
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}"
            )
        self.model = Model(model_path)
        self.temp_dir = Path(tempfile.gettempdir()) / "telegram_voice"
        self.temp_dir.mkdir(exist_ok=True)

    async def convert_to_wav(self, input_file: str, output_file: str):
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            input_file,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            output_file,
            "-y",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await process.communicate()

    async def transcribe_voice(self, voice_file: str) -> str:
        try:
            wav_file = str(self.temp_dir / f"{os.path.basename(voice_file)}.wav")
            await self.convert_to_wav(voice_file, wav_file)

            wf = wave.open(wav_file, "rb")
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part = json.loads(rec.Result())
                    if "text" in part and part["text"]:
                        results.append(part["text"])

            part = json.loads(rec.FinalResult())
            if "text" in part and part["text"]:
                results.append(part["text"])

            wf.close()
            os.remove(voice_file)
            os.remove(wav_file)

            return " ".join(results)
        except Exception as e:
            logger.error(f"Error in voice transcription: {e}")
            raise


class ClaudeChatBot:
    def __init__(
        self,
        name: str,
        api_id: int,
        api_hash: str,
        bot_token: str,
        claude_api_key: str,
        context_file: str = "prompts/context.txt",
    ):

        self.current_manager_index = 0
        self.managers = [7669573911, 6371165930]  # список ID менеджеров

        self.app = Client(name, api_id=api_id, api_hash=api_hash, bot_token=bot_token)

        self.claude_client = AsyncAnthropic(api_key=claude_api_key)
        self.conversations_lock = asyncio.Lock()
        self.conversations = {}
        self.context_file = context_file
        self.api_semaphore = asyncio.Semaphore(3)

        self.app.on_message(filters.command("start"))(self.start_command)
        self.app.on_message(filters.text & ~filters.command("start"))(
            self.process_user_message
        )
        self.app.on_callback_query()(self.process_callback)

        self.voice_processor = VoiceProcessor()
        self.app.on_message(filters.voice)(self.handle_voice_message)

    async def run(self):
        await self.app.start()
        await idle()

    async def handle_voice_message(self, client: Client, message: Message):
        """Handle voice messages"""
        user_id = message.from_user.id
        user_name = message.from_user.username

        if user_id not in self.conversations:
            return

        conv = self.conversations[user_id]

        if conv.processing_lock.locked():
            try:
                await message.reply_text(
                    "⏳ Пожалуйста, дождитесь обработки предыдущего сообщения.",
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
            return

        async with conv.processing_lock:
            try:
                # Download voice message
                temp_file = str(
                    self.voice_processor.temp_dir
                    / f"voice_{user_id}_{datetime.now().timestamp()}.oga"
                )
                await message.download(temp_file)

                # Transcribe
                text = await self.voice_processor.transcribe_voice(temp_file)

                if text:
                    # Process as regular message
                    async with conv.message_lock:
                        if conv.is_processing:
                            await message.reply_text(
                                "⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение.",
                            )
                            return

                        conv.is_processing = True
                        conv.last_message_id = message.id
                        conv.last_message_time = datetime.now()

                        # Add transcribed message to conversation
                        conv.messages.append(
                            {
                                "role": "user",
                                "content": text,
                                "timestamp": datetime.now().isoformat(),
                                "message_id": message.id,
                            }
                        )

                        if conv.chat_history:
                            conv.chat_history.add_message("user", text)

                        # Get Claude response
                        async with self.api_semaphore:
                            response = await self.get_claude_response(user_id)

                        # Handle response
                        await message.reply_text(
                            response,
                            reply_markup=(
                                InlineKeyboardMarkup(
                                    [
                                        [
                                            InlineKeyboardButton(
                                                "⬅️ Окончить разговор",
                                                callback_data="end_conversation",
                                            )
                                        ]
                                    ]
                                )
                                if "завершить диалог" in response.lower()
                                else None
                            ),
                        )

                        if "https://t.me/+4XNh7O7QS-BlNjg6" in response:
                            await self.end_conversation(user_id, user_name)

                else:
                    await message.reply_text(
                        "Извините, не удалось распознать голосовое сообщение."
                    )

            except Exception as e:
                logger.error(f"Voice message processing error: {e}")
                await message.reply_text(
                    "🚫 Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз."
                )
            finally:
                conv.is_processing = False

    async def start_command(self, client: Client, message: Message):
        try:
            if message.from_user.id in self.conversations:
                await message.reply_text(
                    "У вас уже есть активный диалог. Хотите начать новый?",
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton(
                                    "Начать диалог ➡️", callback_data="initialize_chat"
                                )
                            ]
                        ]
                    ),
                )
                return

            sent_message = await message.reply_text(
                """Привет! 👋\n
Я - AI Димы Иванова, обученный на сотнях Гб его контента, включая приватные материалы. Помогу разобраться, подходит ли тебе наша инвестиционная стратегия 🤝\n
Интересует, как на самом деле работает крипта? Отлично - первый шаг уже сделан. Здесь анализируют потоки капитала и видят реальные механизмы рынка крипты 🎯 \n

Задам несколько ключевых вопросов, чтобы понять, найдется ли место в нашем клубе успешных инвесторов именно для тебя. Это займет пару минут. \n
Жми «Начать диалог» когда будешь готов ⬇️\n

🍀 Во благо 🍀""",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Начать диалог ➡️", callback_data="initialize_chat"
                            )
                        ]
                    ]
                ),
            )

            async with self.conversations_lock:
                self.conversations[message.from_user.id] = UserConversation(
                    message.from_user.id
                )
                self.conversations[message.from_user.id].start_message_id = (
                    sent_message.id
                )

        except Exception as e:
            await message.reply_text(
                "Произошла ошибка при запуске. Пожалуйста, попробуйте позже."
            )

    async def _show_typing_status(self, user_id: int):
        """Show typing status while waiting for Claude response"""
        try:
            while True:
                try:
                    await self.app.send_chat_action(
                        chat_id=user_id, action=ChatAction.TYPING
                    )
                    await asyncio.sleep(4)  # Refresh every 4 seconds
                except pyrogram.errors.FloodWait as e:
                    logger.warning(
                        f"FloodWait in typing status for user {user_id}: {e.value} seconds"
                    )
                    await asyncio.sleep(e.value)
                except Exception as e:
                    logger.error(f"Error in typing status for user {user_id}: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Typing status task cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"Unexpected error in typing status for user {user_id}: {e}")

    async def process_callback(self, client: Client, callback_query: CallbackQuery):
        """Handle callback queries"""
        try:
            if callback_query.data == "initialize_chat":
                # Удаляем только кнопку, оставляя приветственное сообщение
                await callback_query.message.edit_reply_markup(reply_markup=None)
                await self.initialize_conversation(callback_query.from_user.id)
                await callback_query.answer("Диалог начат!")

            elif callback_query.data == "end_conversation":
                # Удаляем только кнопку с последнего сообщения
                await callback_query.message.edit_reply_markup(reply_markup=None)
                await callback_query.answer()

                if not self.conversations.get(callback_query.from_user.id):
                    return

                if self.conversations[callback_query.from_user.id].conversation_active:
                    await callback_query.message.reply_text("""
🌟Было приятно общаться с вами. Вот ссылки на наши ресурсы:
Telegram: https://t.me/+4XNh7O7QS-BlNjg6  
Instagram: https://www.instagram.com/dimitriy.live_/  
Youtube: https://youtube.com/@schastlivyi_investor?si=VgN5Bjr-rNRKEdHC
                    """)
                    # End the conversation

                    await self.end_conversation(
                        callback_query.from_user.id, callback_query.from_user.username
                    )

        except Exception as e:
            logger.error(f"Error processing callback: {e}\n{traceback.format_exc()}")
            await callback_query.answer(
                "Произошла ошибка. Пожалуйста, попробуйте позже."
            )

    async def process_user_message(self, client: Client, message: Message):
        user_id = message.from_user.id
        username = message.from_user.username

        async with self.conversations_lock:
            if user_id not in self.conversations:
                return
            conv = self.conversations[user_id]

        async with conv.processing_lock:
            async with conv.message_lock:
                try:
                    conv.is_processing = True
                    conv.last_message_id = message.id
                    conv.last_message_time = datetime.now()

                    conv.messages.append(
                        {
                            "role": "user",
                            "content": message.text,
                            "timestamp": datetime.now().isoformat(),
                            "message_id": message.id,
                        }
                    )

                    conv.chat_history.add_message("user", message.text)

                    async with self.api_semaphore:
                        response = await self.get_claude_response(user_id)

                    sent_message = await message.reply_text(
                        response,
                        reply_markup=(
                            InlineKeyboardMarkup(
                                [
                                    [
                                        InlineKeyboardButton(
                                            "⬅️ Окончить разговор",
                                            callback_data="end_conversation",
                                        )
                                    ]
                                ]
                            )
                            if "завершить диалог" in response.lower()
                            else None
                        ),
                    )
                    conv.last_bot_message = sent_message

                    if "https://t.me/+4XNh7O7QS-BlNjg6" in response:
                        await self.end_conversation(user_id, username)

                finally:
                    conv.is_processing = False

    async def get_claude_response(self, user_id: int) -> str:
        if user_id not in self.conversations:
            raise ValueError(f"No active conversation for user {user_id}")

        conv = self.conversations[user_id]
        messages = []

        if conv.system_prompt:
            messages.append({"role": "assistant", "content": conv.system_prompt})
            if not conv.chat_history.messages:
                conv.chat_history.add_message("assistant", conv.system_prompt)

        for msg in conv.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})

        typing_task = asyncio.create_task(self._show_typing_status(user_id))

        try:
            response = await self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                temperature=0.7,
            )

            claude_message = response.content[0].text

            try:
                response_data = json.loads(claude_message)
                claude_message = response_data.get("Reply", claude_message)
            except:
                pass

            conv.messages.append(
                {
                    "role": "assistant",
                    "content": claude_message,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            conv.chat_history.add_message("assistant", claude_message)

            return claude_message

        except Exception as e:
            logger.error(f"Error in get_claude_response: {str(e)}")
            return "Произошла системная ошибка. Пожалуйста, попробуйте позже или начните новый диалог через /start"

        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    async def send_inactivity_prompt(self, user_id: int):
        """Send inactivity prompt with proper state handling"""
        if user_id not in self.conversations:
            logger.warning(
                f"Attempted to send prompt to non-existent conversation {user_id}"
            )
            return

        conv = self.conversations[user_id]

        try:
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🛑 Остановить чат", callback_data="stop_conversation"
                        ),
                        InlineKeyboardButton(
                            "🔄 Продолжить чат", callback_data="resume_conversation"
                        ),
                    ]
                ]
            )

            await self.app.send_message(
                chat_id=user_id,
                text="⏰ Вы отсутствовали в чате 5 часов. Что вы хотите сделать?",
                reply_markup=keyboard,
            )
            logger.info(f"Inactivity prompt sent to user {user_id}")

        except Exception as e:
            logger.error(f"Error sending inactivity prompt to {user_id}: {e}")
            raise
        finally:
            # Сбрасываем флаг обработки в любом случае
            async with conv.message_lock:
                conv.is_processing = False

    async def inactivity_timer(self, user_id: int):
        """Manage user inactivity with proper state handling"""
        try:
            while True:
                # Проверяем существование беседы без блокировок
                if user_id not in self.conversations:
                    logger.info(
                        f"Timer stopped: conversation {user_id} no longer exists"
                    )
                    return

                conv = self.conversations[user_id]
                current_time = datetime.now()

                # Используем только message_lock для атомарных операций
                async with conv.message_lock:
                    if not conv.last_message_time:
                        logger.warning(f"No last message time for user {user_id}")
                        return

                    time_diff = current_time - conv.last_message_time

                    # Если прошло больше 5 часов и не отправлено предупреждение
                    if time_diff > timedelta(hours=5) and not conv.is_processing:
                        logger.info(f"Sending inactivity prompt to user {user_id}")
                        conv.is_processing = True

                        try:
                            # Отправляем предупреждение вне блокировки
                            await self.send_inactivity_prompt(user_id)
                            # Обновляем время последнего сообщения
                            conv.last_message_time = current_time
                        except Exception as e:
                            logger.error(f"Failed to send inactivity prompt: {e}")
                        finally:
                            conv.is_processing = False

                # Ждем вне блокировок
                await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.info(f"Timer cancelled for user {user_id}")
            raise

        except Exception as e:
            logger.error(f"Timer error for user {user_id}: {e}")

    async def load_rules(self) -> str:
        """Load rules from file"""
        try:
            async with aiofiles.open("prompts/rules.txt", "r", encoding="utf-8") as f:
                content = await f.read()
                if not content.strip():
                    raise ValueError("Rules file is empty")
                return content
        except Exception as e:
            logger.error(f"Error loading rules file: {e}")
            return ""

    async def load_context(self) -> str:
        """Load context from file"""
        try:
            async with aiofiles.open(self.context_file, "r", encoding="utf-8") as f:
                content = await f.read()
                if not content.strip():
                    raise ValueError("Context file is empty")
                return content
        except Exception as e:
            logger.error(f"Error loading context file: {e}")
            return ""

    async def initialize_conversation(self, user_id: int):
        async with self.conversations_lock:
            try:
                context = await self.load_context()
                rules = await self.load_rules()

                conv = UserConversation(user_id=user_id)
                self.conversations[user_id] = conv

                conv.system_prompt = f"""
Контекст:
{context}

Правила:
{rules}
"""
                conv.messages = [
                    {
                        "role": "assistant",
                        "content": conv.system_prompt,
                        "timestamp": datetime.now().isoformat(),
                    },
                    {
                        "role": "user",
                        "content": "Без лишинх приветствий - давай начнём наш опрос.",
                        "timestamp": datetime.now().isoformat(),
                    },
                ]

                response_default = """
Чтобы наше общение было максимально продуктивным - со мной нужно общаться так, как с самим Димой. Я создан, чтобы продолжать его дело 🤖💫\n
И да, мне можно отправлять голосовые сообщения и задавать любые вопросы - \n  
но для начала - как мне к вам обращаться и что привело вас к нам?
                """

                self.conversations[user_id].chat_history.add_message("assistant", response_default)

                sent_message = await self.app.send_message(
                    chat_id=user_id,
                    text=response_default,
                )
                conv.last_bot_message = sent_message

                conv.timer_task = asyncio.create_task(self.inactivity_timer(user_id))

            except Exception as e:
                await self.app.send_message(
                    chat_id=user_id, text="🚫 Ошибка инициализации. Попробуйте позже."
                )

    async def end_conversation(self, user_id: int, username: str = None):
        ADMIN_IDS = [339041653, 6740195967, 446842625]
        print("Ending for", user_id)

        # process manager reporting logic: 2 managers - implementing counter to send c%i==0 to manager A, and c%i==1 to manager B
        manager_id = self.managers[self.current_manager_index]
        self.current_manager_index = (self.current_manager_index + 1) % len(
            self.managers
        )
        ADMIN_IDS += [manager_id]

        await self.app.send_message(
            chat_id=user_id,
            text="Спасибо за общение! Ваша заявка будет рассмотрена в ближайшее время.",
        )
        try:
            async with self.conversations_lock:
                if user_id not in self.conversations:
                    return

                conv: UserConversation = self.conversations[user_id]
                conv.conversation_active = False

                if conv.timer_task and not conv.timer_task.done():
                    conv.timer_task.cancel()
                    try:
                        await conv.timer_task
                    except:
                        pass

                if conv.chat_history:
                    history_file = conv.chat_history.save_to_file()
                    try:
                        for admin_id in ADMIN_IDS:
                            await self.app.send_document(
                                chat_id=admin_id,
                                document=history_file,
                                caption=f"История диалога с пользователем @{username} {user_id}",
                            )
                    except:
                        pass

                    analyzer = DialogueAnalyzer(self.claude_client)
                    report, report_file = await analyzer.create_report(
                        history_file, str(user_id)
                    )
                    logger.info(
                        f"Report for user {user_id} @{username}:\n{report_file}"
                    )
                    if report:
                        logger.info(f"Report for user {user_id}:  {report[:20]}")
                        match = (
                            re.search(
                                r"### Финансовый профиль.*?(?=\n###)", report, re.S
                            ).group(0)
                            or "Смотри в отчете"
                        )
                        for admin_id in ADMIN_IDS:
                            try:
                                await self.app.send_document(
                                    chat_id=admin_id,
                                    document=report_file,
                                    caption=f"Анализ диалога с пользователем {user_id}; TG: @{username}\n"
                                    f"Финансовый профиль:\n"
                                    f"{match}\n\n",
                                )
                            except Exception as e:
                                logger.error(
                                    f"Send document failed for admin {admin_id}: {e}"
                                )
                                pass

                del self.conversations[user_id]

        except Exception as e:
            logger.error(f"Error ending conversation: {e}")


async def main():
    import os

    from dotenv import load_dotenv

    load_dotenv()

    bot = ClaudeChatBot(
        name="claude_bot",
        api_id=int(os.environ.get("API_ID")),
        api_hash=os.environ.get("API_HASH"),
        bot_token=" ",
        claude_api_key=os.environ.get("CLAUDE_API_KEY"),
    )

    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
