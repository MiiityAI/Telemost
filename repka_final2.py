import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import aiofiles
import os
import json
import traceback
import signal
import wave
import subprocess
import tempfile
from pathlib import Path
from vosk import Model, KaldiRecognizer

from pyrogram import Client, filters, idle
from pyrogram.types import (
    InlineKeyboardMarkup, 
    InlineKeyboardButton, 
    Message, 
    CallbackQuery
)
from anthropic import Anthropic
import httpx
import pyrogram
from pyrogram.enums import ChatAction

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        self.filename = f"{user_id}_{datetime.now().strftime('%H_%M_%d_%m_%y')}.md"

    def add_message(self, role: str, content: str):
        """Add message to history, extracting only text content"""
        if role == "assistant":
            try:
                # Parse JSON response to get only the Reply content
                data = json.loads(content)
                content = data.get("Reply", "")
            except json.JSONDecodeError:
                pass
                
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def save_to_file(self) -> str:
        """Save chat history to file with formatted timestamp"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            for msg in self.messages:
                f.write(f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content']}\n")
                f.write("-" * 80 + "\n")
        
        return self.filename
    
    def get_conversation_text(self) -> str:
        """Get clean conversation text for analysis"""
        conversation = []
        for msg in self.messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation.append(f"{role}: {msg['content']}")
        return "\n".join(conversation)
    
    @property
    def duration(self) -> str:
        """Get conversation duration in readable format"""
        duration = datetime.now() - self.start_time
        minutes = duration.seconds // 60
        seconds = duration.seconds % 60
        return f"{minutes}m {seconds}s"
    
    def __str__(self) -> str:
        """String representation of chat history"""
        return (
            f"Chat History for User {self.user_id}\n"
            f"Duration: {self.duration}\n"
            f"Messages: {len(self.messages)}\n"
            f"Status: Completed"
        )

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
    last_message: Optional[str] = None
    last_response: Optional[str] = None
    is_processing: bool = False
    chat_history: Optional[AIChatHistory] = None
    last_message_id: Optional[int] = None
    farewell_pending: bool = False
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
            if self.message_lock.locked():
                logger.info(f"[CLEANUP-LOCK] Force releasing message_lock for {self.user_id}")
                self.message_lock.release()
                logger.info(f"[CLEANUP-LOCK] Message lock released for {self.user_id}")
                
            if self.timer_task and not self.timer_task.done():
                logger.info(f"[CLEANUP-TIMER] Cancelling timer task for {self.user_id}")
                self.timer_task.cancel()
                try:
                    await asyncio.wait_for(self.timer_task, timeout=1.0)
                    logger.info(f"[CLEANUP-TIMER] Timer cancelled successfully for {self.user_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"[CLEANUP-TIMER] Timer cancellation timeout for {self.user_id}")
                except asyncio.CancelledError:
                    logger.info(f"[CLEANUP-TIMER] Timer cancelled for {self.user_id}")
                    
            history_file = None
            if self.chat_history:
                logger.info(f"[CLEANUP-HISTORY] Saving history for {self.user_id}")
                history_file = self.chat_history.save_to_file()
                logger.info(f"[CLEANUP-HISTORY] History saved to {history_file} for {self.user_id}")
                
            return history_file
            
        except Exception as e:
            logger.error(f"[CLEANUP-ERROR] Error in cleanup for {self.user_id}: {e}\n{traceback.format_exc()}")
            return None
        finally:
            logger.info(f"[CLEANUP-END] Cleanup completed for {self.user_id}")

class DialogueAnalyzer:
    def __init__(self, anthropic_client):
        self.claude_client = anthropic_client
        
    async def create_report(self, chat_history_path: str, user_id: str) -> Optional[str]:
        """Asynchronous function to create analysis report"""
        logger.info(f"Starting report creation for user {user_id}")
        
        try:
            # Read criteria template
            try:
                logger.info("Attempting to read criteria template")
                async with aiofiles.open('estimated.md', 'r', encoding='utf-8') as f:
                    criteria = await f.read()
                    if not criteria.strip():
                        logger.error("Criteria template is empty")
                        raise ValueError("Criteria template is empty")
                    logger.info("Criteria template loaded successfully")
            except FileNotFoundError:
                logger.error("Criteria template file 'estimated.md' not found")
                raise
            except Exception as e:
                logger.error(f"Failed to load criteria template: {e}\n{traceback.format_exc()}")
                raise

            # Read chat history
            try:
                logger.info(f"Attempting to read chat history from {chat_history_path}")
                async with aiofiles.open(chat_history_path, 'r', encoding='utf-8') as f:
                    dialogue = await f.read()
                    if not dialogue.strip():
                        logger.error("Chat history is empty")
                        raise ValueError("Chat history is empty")
                    logger.info(f"Chat history loaded successfully: {len(dialogue)} characters")
            except FileNotFoundError:
                logger.error(f"Chat history file not found: {chat_history_path}")
                raise
            except Exception as e:
                logger.error(f"Failed to load chat history: {e}\n{traceback.format_exc()}")
                raise

            # Prepare analysis prompt with updated structure
            prompt = f"""
You are an experienced HR analyst evaluating a candidate based on their dialogue. Create a structured report in markdown format.

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

Important: 
- Keep the format consistent with section headers
- Be specific and concise
- Base all conclusions on dialogue evidence
"""
            logger.info("[REPORT] Sending request to Claude API")

            try:
                response = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                report = response.content[0].text
                logger.info(f"[REPORT] Received response ({len(report)} chars)")
                
                # Updated validation without summary requirement
                required_keywords = ['evaluation', 'score', 'recommendation', 'strength']
                found_keywords = [keyword for keyword in required_keywords 
                                if keyword.lower() in report.lower()]
                
                if len(found_keywords) < 3:  # Allow some flexibility
                    logger.warning(f"[REPORT] Report may be incomplete. Found sections: {found_keywords}")
                    report = "# Analysis Report\n\n" + report
                    
                # Save report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"{user_id}_report_{timestamp}.md"
                async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
                    await f.write(f"# Analysis Report for User {user_id}\n")
                    await f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    await f.write(report)
                
                logger.info(f"[REPORT] Report saved to {report_path}")
                return report
                
            except Exception as e:
                logger.error(f"[REPORT] Claude API error: {e}")
                return f"⚠️ Error generating report: {str(e)}"

            # Save report with timestamp
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = f"{user_id}_report_{timestamp}.md"
                logger.info(f"Saving report to {report_path}")
                async with aiofiles.open(report_path, 'w', encoding='utf-8') as f:
                    await f.write(f"# Analysis Report for User {user_id}\n")
                    await f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    await f.write(report)
                logger.info(f"Report saved successfully to {report_path}")
                
            except Exception as e:
                logger.error(f"Failed to save report: {e}\n{traceback.format_exc()}")
                # Continue execution as report will still be sent to group
            
            logger.info("Returning generated report")
            return report
            
        except Exception as e:
            error_msg = f"Error in create_report: {str(e)}\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            return None

class VoiceProcessor:
    def __init__(self, model_path: str = "./vosk-model-ru"):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}")
        self.model = Model(model_path)
        self.temp_dir = Path(tempfile.gettempdir()) / 'telegram_voice'
        self.temp_dir.mkdir(exist_ok=True)

    def convert_to_wav(self, input_file: str, output_file: str):
        subprocess.run([
            'ffmpeg', '-i', input_file,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_file,
            '-y'
        ], check=True)

    async def transcribe_voice(self, voice_file: str) -> str:
        try:
            wav_file = str(self.temp_dir / f"{os.path.basename(voice_file)}.wav")
            self.convert_to_wav(voice_file, wav_file)

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
                    if 'text' in part and part['text']:
                        results.append(part['text'])

            part = json.loads(rec.FinalResult())
            if 'text' in part and part['text']:
                results.append(part['text'])

            wf.close()
            os.remove(voice_file)
            os.remove(wav_file)

            return ' '.join(results)
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
        context_file: str = "context.txt"
    ):
        """Initialize bot with required credentials and settings"""
        # Инициализация Pyrogram клиента
        self.app = Client(
            name,
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token
        )
        
        # Инициализация Claude клиента
        self.claude_client = Anthropic(api_key=claude_api_key)
        
        # Инициализация блокировок и семафоров
        self.conversations_lock = asyncio.Lock()
        self.global_lock = asyncio.Lock()
        self.timer_lock = asyncio.Lock()
        self.api_semaphore = asyncio.Semaphore(3)
        
        # Инициализация остальных атрибутов
        self.conversations = {}
        self.context_file = context_file
        
        # Регистрация обработчиков
        self.app.on_message(filters.command("start"))(self.start_command)
        self.app.on_message(filters.text & ~filters.command("start"))(self.process_user_message)
        self.app.on_callback_query()(self.process_callback)
        
        # Enhanced queue management
        self.claude_queue = asyncio.Queue()
        self.claude_worker_task = None
        self.last_claude_request_time = None
        self.claude_request_lock = asyncio.Lock()
        self.active_requests = set()  # Track active requests by user_id
        self.request_lock = asyncio.Lock()  # Lock for request management
        
        # Add voice processor
        self.voice_processor = VoiceProcessor()
        
        # Add voice message handler
        self.app.on_message(filters.voice)(self.handle_voice_message)
        
        logger.info("Bot initialized successfully with all required components")

    async def run(self):
        """Start the bot"""
        try:
            logger.info("Starting bot...")
            # Start Claude request worker
            self.claude_worker_task = asyncio.create_task(self.process_claude_queue())
            await self.app.start()
            logger.info("Bot started successfully")
            await idle()
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
        finally:
            logger.info("Stopping bot...")
            if self.claude_worker_task:
                self.claude_worker_task.cancel()
            await self.app.stop()

    async def start_command(self, client: Client, message: Message):
        """Handle /start command"""
        try:
            # Проверяем, существует ли уже активная беседа
            if message.from_user.id in self.conversations:
                await message.reply_text(
                    "У вас уже есть активный диалог. Хотите начать новый?",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Начать диалог ➡️", callback_data="initialize_chat")]
                    ])
                )
                return

            sent_message = await message.reply_text("""Привет👋 Меня зовут Дмитрий.\n
Да, я AI, но я умнее всех чатов GPT, которые вы видели, так как я создан и обучен на сотнях Гб текстов, видео и аудиосообщений, авторами которых является Дмитрий Иванов.\n
Я задам вам несколько вопросов, которые помогут сформировать заявку, а также обработаю и отвечу на ваши вопросы так, как это сделал бы сам Дмитрий.\n
Как будете готовы, нажимайте кнопку «Оставить заявку», это займет не более 2ух минут.\n
                                  🍀 Во благо 🍀""",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Начать диалог ➡️", callback_data="initialize_chat")]
                ])
            )
            # Сохраняем message_id для последующего редактирования
            if message.from_user.id not in self.conversations:
                self.conversations[message.from_user.id] = UserConversation(message.from_user.id)
            self.conversations[message.from_user.id].start_message_id = sent_message.id
            
            logger.info(f"Start command processed for user {message.from_user.id}")
            
        except Exception as e:
            logger.error(f"Error processing start command: {e}")
            await message.reply_text(
                "Произошла ошибка при запуске. Пожалуйста, попробуйте позже."
            )

    async def process_internal_signals(self, user_id: int, message: Message, claude_message: str) -> bool:
        """Process signals from Claude's response."""
        try:
            response_data = json.loads(claude_message)
            return False
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in process_internal_signals: {str(e)}\nInput: {claude_message[:200]}...")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in process_internal_signals: {str(e)}\n{traceback.format_exc()}")
            return False

    async def process_user_message(self, client: Client, message: Message):
        """Process incoming user messages with proper synchronization"""
        user_id = message.from_user.id
        
        async with self.conversations_lock:
            if user_id not in self.conversations:
                return

            conv = self.conversations[user_id]
        
        async with conv.processing_lock:
            async with conv.message_lock:
                try:
                    if message.id == conv.last_message_id:
                        logger.info(f"Duplicate message detected for user {user_id}")
                        return
                    
                    if conv.is_processing:
                        try:
                            await message.delete()
                            await message.reply_text(
                                "⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение.",
                                delete_after=3
                            )
                        except Exception as e:
                            logger.error(f"Failed to delete message: {e}")
                        return
                    
                    # Если есть предыдущее сообщение с кнопкой, удаляем кнопку
                    if hasattr(conv, 'last_bot_message') and conv.last_bot_message:
                        try:
                            await conv.last_bot_message.edit_reply_markup(reply_markup=None)
                        except Exception as e:
                            logger.warning(f"Could not remove previous button: {e}")
                    
                    conv.is_processing = True
                    conv.last_message_id = message.id
                    conv.last_message_time = datetime.now()
                    
                    # Process message
                    conv.messages.append({
                        "role": "user",
                        "content": message.text,
                        "timestamp": datetime.now().isoformat(),
                        "message_id": message.id
                    })
                    
                    if conv.chat_history:
                        conv.chat_history.add_message("user", message.text)
                    
                    async with self.api_semaphore:
                        response = await self.get_claude_response(user_id)
                    
                    # Send response with end conversation button and store the message
                    sent_message = await message.reply_text(
                        response,
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton("⬅️ Окончить разговор", callback_data="end_conversation")
                        ]])
                    )
                    conv.last_bot_message = sent_message
                    
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    error_message = await message.reply_text(
                        "🚫 Произошла ошибка. Пожалуйста, попробуйте еще раз.",
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton("⬅️ Окончить разговор", callback_data="end_conversation")
                        ]])
                    )
                    conv.last_bot_message = error_message
                finally:
                    conv.is_processing = False

    async def process_claude_queue(self):
        """Process Claude API requests queue with enhanced rate limiting and error handling"""
        while True:
            try:
                request_data = await self.claude_queue.get()
                user_id, messages, future = request_data
                
                logger.info(f"[QUEUE] Processing request from user {user_id}")
                
                async with self.claude_request_lock:
                    if self.last_claude_request_time:
                        elapsed = datetime.now() - self.last_claude_request_time
                        if elapsed.total_seconds() < 10:
                            wait_time = 10 - elapsed.total_seconds()
                            logger.info(f"[QUEUE] Waiting {wait_time:.2f} seconds before processing request from user {user_id}")
                            await asyncio.sleep(wait_time)
                    
                    try:
                        logger.info(f"[QUEUE] Sending Claude API request for user {user_id}")
                        response = await asyncio.to_thread(
                            self.claude_client.messages.create,
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1000,
                            messages=messages
                        )
                        
                        logger.info(f"[CLAUDE-RESPONSE] User: {user_id}\nResponse: {response.content[0].text}\n{'-'*80}")
                        future.set_result(response)
                        
                    except httpx.TimeoutException as e:
                        logger.error(f"[CLAUDE-ERROR] Timeout error for user {user_id}: {str(e)}")
                        future.set_exception(e)
                    except httpx.HTTPStatusError as e:
                        logger.error(f"[CLAUDE-ERROR] HTTP error for user {user_id}: {str(e)}")
                        future.set_exception(e)
                    except Exception as e:
                        logger.error(f"[CLAUDE-ERROR] Unexpected error for user {user_id}: {str(e)}\n{traceback.format_exc()}")
                        future.set_exception(e)
                    finally:
                        self.last_claude_request_time = datetime.now()
                        self.claude_queue.task_done()
                        logger.info(f"[QUEUE] Completed request processing for user {user_id}")
                        
            except asyncio.CancelledError:
                logger.info("[QUEUE] Claude queue processor shutting down")
                break
            except Exception as e:
                logger.error(f"[QUEUE] Critical error in queue processor: {str(e)}\n{traceback.format_exc()}")
                await asyncio.sleep(1)

    async def get_claude_response(self, user_id: int) -> str:
        """Get response from Claude using enhanced queue system"""
        if user_id not in self.conversations:
            raise ValueError(f"No active conversation for user {user_id}")
        
        typing_task = None
        try:
            conv = self.conversations[user_id]
            messages = []
            
            # Add system prompt if exists
            if conv.system_prompt:
                messages.append({"role": "assistant", "content": conv.system_prompt})
                if conv.chat_history and not conv.chat_history.messages:
                    conv.chat_history.add_message("assistant", conv.system_prompt)
            
            # Add conversation history
            for msg in conv.messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Start typing indicator
            typing_task = asyncio.create_task(self._show_typing_status(user_id))
            
            try:
                # Direct API call with error handling
                response = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    temperature=0.7
                )
                self.last_claude_request_time = datetime.now()
                
                claude_message = response.content[0].text
                
                # Extract only the "Reply" text from JSON response
                try:
                    response_data = json.loads(claude_message)
                    claude_message = response_data.get("Reply", claude_message)
                except:
                    # If JSON parsing fails, use the raw message
                    pass
                
                # Store message in conversation history
                conv.messages.append({
                    "role": "assistant",
                    "content": claude_message,
                    "timestamp": datetime.now().isoformat()
                })
                
                if conv.chat_history:
                    conv.chat_history.add_message("assistant", claude_message)
                
                return claude_message
                
            except Exception as e:
                logger.error(f"Error processing Claude response: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in get_claude_response for user {user_id}: {str(e)}\n{traceback.format_exc()}")
            return "Произошла системная ошибка. Пожалуйста, попробуйте позже или начните новый диалог через /start"
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
        
            async with self.request_lock:
                self.active_requests.discard(user_id)

    def _format_conversation_history(self, messages: list) -> str:
        """Format conversation history for context"""
        formatted_history = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Extract only the "Reply" part if it's a JSON response from assistant
            content = msg["content"]
            if role == "Assistant" and content.startswith("{"):
                try:
                    content = json.loads(content)["Reply"]
                except (json.JSONDecodeError, KeyError):
                    pass
            formatted_history.append(f"{role}: {content}")
        return "\n".join(formatted_history)

    async def inactivity_timer(self, user_id: int):
        """Manage user inactivity with proper state handling"""
        try:
            while True:
                # Проверяем существование беседы без блокировок
                if user_id not in self.conversations:
                    logger.info(f"Timer stopped: conversation {user_id} no longer exists")
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

    async def send_inactivity_prompt(self, user_id: int):
        """Send inactivity prompt with proper state handling"""
        if user_id not in self.conversations:
            logger.warning(f"Attempted to send prompt to non-existent conversation {user_id}")
            return
        
        conv = self.conversations[user_id]
        
        try:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🛑 Остановить чат", callback_data="stop_conversation"),
                    InlineKeyboardButton("🔄 Продолжить чат", callback_data="resume_conversation")
                ]
            ])

            await self.app.send_message(
                chat_id=user_id,
                text="⏰ Вы отсутствовали в чате 5 часов. Что вы хотите сделать?",
                reply_markup=keyboard
            )
            logger.info(f"Inactivity prompt sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error sending inactivity prompt to {user_id}: {e}")
            raise
        finally:
            # Сбрасываем флаг обработки в любом случае
            async with conv.message_lock:
                conv.is_processing = False

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
                # End the conversation
                await self.end_conversation(callback_query.from_user.id)
                await callback_query.answer("Диалог завершен!")
                
        except Exception as e:
            logger.error(f"Error processing callback: {e}\n{traceback.format_exc()}")
            await callback_query.answer("Произошла ошибка. Пожалуйста, попробуйте позже.")

    async def initialize_conversation(self, user_id: int):
        """Initialize Claude chat with proper synchronization"""
        async with self.global_lock:
            try:
                # Load context and rules
                context = await self.load_context()
                rules = await self.load_rules()
                
                if not context or not rules:
                    raise ValueError("Failed to load context or rules")
                
                # Create new conversation with explicit chat history initialization
                conv = UserConversation(user_id=user_id)
                if not conv.chat_history:  # Добавляем дополнительную проверку
                    conv.chat_history = AIChatHistory(user_id)
                    logger.info(f"Chat history explicitly initialized for user {user_id}")
                
                self.conversations[user_id] = conv
                
                # Set system prompt
                conv.system_prompt = f"""
Контекст:
{context}

Правила:
{rules}
"""
                # Initialize conversation with system prompt
                conv.messages = [
                    {"role": "assistant", "content": conv.system_prompt, "timestamp": datetime.now().isoformat()},
                    {"role": "user", "content": "Без лишинх приветствий - давай начнём наш опрос.", "timestamp": datetime.now().isoformat()}
                ]
                
                async with self.api_semaphore:
                    response = await self.get_claude_response(user_id)
                
                # Send initial message with end conversation button
                sent_message = await self.app.send_message(
                    chat_id=user_id,
                    text=response,
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("⬅️ Окончить разговор", callback_data="end_conversation")
                    ]])
                )
                conv.last_bot_message = sent_message
                
                # Set initialization flag
                async with conv.message_lock:
                    conv.is_initialized = True
                
                # Create timer
                async with self.timer_lock:
                    conv.timer_task = asyncio.create_task(self.inactivity_timer(user_id))
                
            except Exception as e:
                logger.error(f"Initialization error: {e}")
                await self.app.send_message(
                    chat_id=user_id,
                    text="🚫 Ошибка инициализации. Попробуйте позже.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("⬅️ Окончить разговор", callback_data="end_conversation")
                    ]])
                )

    async def load_rules(self) -> str:
        """Load rules from file"""
        try:
            async with aiofiles.open("rules.txt", 'r', encoding='utf-8') as f:
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
            async with aiofiles.open(self.context_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                if not content.strip():
                    raise ValueError("Context file is empty")
                return content
        except Exception as e:
            logger.error(f"Error loading context file: {e}")
            return ""

    async def handle_error(self, user_id: int, error: Exception, message: Optional[Message] = None) -> None:
        """Unified error handling"""
        error_msg = f"Error: {str(error)}"
        logger.error(f"Error in conversation {user_id}: {error_msg}")
        
        try:
            if message:
                await message.reply_text(
                    "Извините, произошла ошибка. Пожалуйста, попробуйте еще раз через /start"
                )
        except Exception as e:
            logger.error(f"Error during error handling: {e}")

    async def _show_typing_status(self, user_id: int):
        """Show typing status while waiting for Claude response"""
        try:
            while True:
                try:
                    await self.app.send_chat_action(
                        chat_id=user_id,
                        action=ChatAction.TYPING
                    )
                    await asyncio.sleep(4)  # Refresh every 4 seconds
                except pyrogram.errors.FloodWait as e:
                    logger.warning(f"FloodWait in typing status for user {user_id}: {e.value} seconds")
                    await asyncio.sleep(e.value)
                except Exception as e:
                    logger.error(f"Error in typing status for user {user_id}: {e}")
                    await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info(f"Typing status task cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"Unexpected error in typing status for user {user_id}: {e}")

    async def handle_voice_message(self, client: Client, message: Message):
        """Handle voice messages"""
        user_id = message.from_user.id
        
        if user_id not in self.conversations:
            return

        conv = self.conversations[user_id]
        
        if conv.processing_lock.locked():
            try:
                await message.reply_text(
                    "⏳ Пожалуйста, дождитесь обработки предыдущего сообщения.",
                    delete_after=3
                )
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
            return

        async with conv.processing_lock:
            try:
                # Download voice message
                temp_file = str(self.voice_processor.temp_dir / f"voice_{user_id}_{datetime.now().timestamp()}.oga")
                await message.download(temp_file)
                
                # Transcribe
                text = await self.voice_processor.transcribe_voice(temp_file)
                
                if text:
                    # Process as regular message
                    async with conv.message_lock:
                        if conv.is_processing:
                            await message.reply_text(
                                "⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение.",
                                delete_after=3
                            )
                            return
                        
                        conv.is_processing = True
                        conv.last_message_id = message.id
                        conv.last_message_time = datetime.now()
                        
                        # Add transcribed message to conversation
                        conv.messages.append({
                            "role": "user",
                            "content": text,
                            "timestamp": datetime.now().isoformat(),
                            "message_id": message.id
                        })
                        
                        if conv.chat_history:
                            conv.chat_history.add_message("user", text)
                        
                        # Get Claude response
                        async with self.api_semaphore:
                            response = await self.get_claude_response(user_id)
                        
                        # Handle response
                        await message.reply_text(response)
                        
                else:
                    await message.reply_text("Извините, не удалось распознать голосовое сообщение.")
                    
            except Exception as e:
                logger.error(f"Voice message processing error: {e}")
                await message.reply_text(
                    "🚫 Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз."
                )
            finally:
                conv.is_processing = False

    async def end_conversation(self, user_id: int):
        """Handle conversation ending and analysis"""
        ADMIN_IDS = [370412257, 339041653]  # List of admin IDs
        
        try:
            if user_id not in self.conversations:
                return
            
            conv = self.conversations[user_id]
            
            # Cancel typing status if active
            if hasattr(conv, 'typing_task') and conv.typing_task and not conv.typing_task.done():
                conv.typing_task.cancel()
                try:
                    await conv.typing_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel timer task if active
            if conv.timer_task and not conv.timer_task.done():
                conv.timer_task.cancel()
                try:
                    await conv.timer_task
                except asyncio.CancelledError:
                    pass
            
            # Save conversation history and analyze
            if conv.chat_history:
                try:
                    # Save history file
                    history_file = conv.chat_history.save_to_file()
                    logger.info(f"Chat history saved to {history_file}")
                    
                    # Ensure estimated.md exists
                    if not os.path.exists('estimated.md'):
                        with open('estimated.md', 'w', encoding='utf-8') as f:
                            f.write("""
1. Понимание продукта:
   - Знание криптовалют
   - Понимание инвестиций
   - Осведомленность о рисках

2. Финансовая готовность:
   - Наличие свободных средств
   - Стабильность дохода
   - Готовность к инвестициям

3. Личностные качества:
   - Обучаемость
   - Коммуникабельность
   - Целеустремленность

4. Мотивация:
   - Четкость целей
   - Реалистичность ожиданий
   - Долгосрочные планы
""")
                    
                    # Generate analysis
                    analyzer = DialogueAnalyzer(self.claude_client)
                    report = await analyzer.create_report(history_file, str(user_id))
                    
                    if report:
                        logger.info("Analysis report generated successfully")
                        
                        # Save report locally first
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        report_file = f"report_{user_id}_{timestamp}.txt"
                        with open(report_file, 'w', encoding='utf-8') as f:
                            f.write(f"Analysis Report for user {user_id}\n\n{report}")
                        logger.info(f"Report saved locally to {report_file}")
                        
                        # Try to send to all admins
                        for admin_id in ADMIN_IDS:
                            try:
                                # Send report and history to admin
                                await self.app.send_document(
                                    chat_id=admin_id,
                                    document=report_file,
                                    caption=f"Анализ диалога с пользователем {user_id}"
                                )
                                await self.app.send_document(
                                    chat_id=admin_id,
                                    document=history_file,
                                    caption=f"История диалога с пользователем {user_id}"
                                )
                                logger.info(f"Report and history sent to admin {admin_id}")
                            except Exception as e:
                                logger.warning(f"Could not send report to admin {admin_id}: {e}")
                
                except Exception as e:
                    logger.error(f"Error generating or saving report: {e}\n{traceback.format_exc()}")
            
            # Remove conversation object
            del self.conversations[user_id]
            
            # Send farewell message without button
            await self.app.send_message(
                chat_id=user_id,
                text="Спасибо за общение! Ваша заявка будет рассмотрена в ближайшее время."
            )
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}\n{traceback.format_exc()}")

async def analyze_dialogue(user_id: str, history_path: str):
    analyzer = DialogueAnalyzer(self.claude_client)
    try:
        await analyzer.create_report(history_path, user_id)
    except Exception as e:
        logger.error(f"Failed to analyze dialogue: {e}")

async def main():
    # Credentials
    API_ID = 
    API_HASH = ' '
    BOT_TOKEN = ' ' #rep
    CLAUDE_API_KEY = ' '

    bot = ClaudeChatBot(
        name="claude_bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN,
        claude_api_key=CLAUDE_API_KEY
    )

    async def shutdown(signal, loop):
        """Cleanup function to handle graceful shutdown"""
        logger.info(f"Received exit signal {signal.name}...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        
        # Cancel all running tasks
        [task.cancel() for task in tasks]
        logger.info(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    loop = asyncio.get_event_loop()
    
    # Handle both SIGINT (Ctrl+C) and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        await shutdown(signal.SIGINT, loop)
    except Exception as e:
        logger.error(f"Bot stopped due to error: {e}")
        await shutdown(signal.SIGTERM, loop)
    finally:
        loop.close()
        logger.info("Successfully shutdown the bot")

if __name__ == '__main__':
    asyncio.run(main())
