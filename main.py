import asyncio
import os
import sys
import shutil
import discord
import json
import uuid
import time
import aiohttp
import subprocess
import aiofiles
import chromadb

from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from chromadb import Documents, EmbeddingFunction, Embeddings



# Variables/Configs
if not os.path.exists("configs/.env"):
    subprocess.run([sys.executable, 'gui.py'])


config_file = "configs/config.json"

with open (config_file, "r") as f:
    config_list = json.load(f)

locations_list = config_list["file_locations"]
locations_keys = ["instructions", "instruction_system", "instructions_auxiliary", "history", "memory", "emotion_state"]
instructions_file, instructions_system_file,instructions_auxiliary_file, chat_log_file, memory_file, emotion_state_file = map(locations_list.__getitem__, locations_keys)

id_list = config_list["id_list"]
keys = ["message_channel", "log_channel", "self_id", "owner_id", "self_name"]
message_channel, log_channel_id, self_id, owner_id, self_name = map(id_list.__getitem__, keys)

settings_list = config_list["settings"]
settings_key = ["image_processing", "gemini_model_main", "gemini_model_fallback", "gemini_model_auxiliary", "gemini_model_embedding"]
image_processing, gemini_model_main, gemini_model_fallback, gemini_model_auxiliary, gemini_model_embedding = map(settings_list.__getitem__, settings_key)

name_list = config_list["name_list"]


# Initializing
intents = discord.Intents.all()
load_dotenv("configs/.env")
bot = discord.Bot(intents=intents)


# Gemini variables/initialization
use_fallback_model = False
last_fallback_time = None
gemini_overloaded = False
fallback_timeout = 21600
fallback_timeout_overloaded = 60
gemini_model_current = gemini_model_main

gemini = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

with open(instructions_file, 'r', encoding="utf-8") as file:
    instructions_main = file.read()

with open (instructions_system_file, 'r', encoding="utf-8") as file:
    instructions_system = file.read()

instructions_complete = instructions_main + f"\n\n" + instructions_system

with open(instructions_auxiliary_file, 'r', encoding="utf-8") as file:
    instructions_auxiliary = file.read()


# ChromaDB initialization
chroma = chromadb.PersistentClient(path="data/chroma")

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("Initialized embedding function.")
    # IDK what else to put here, the fucking thing wouldn't leave me alone without init

    def __call__(self, memory: Documents) -> Embeddings:
        result = gemini.models.embed_content(
            model = gemini_model_embedding,
            contents = memory,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )

        embeddings = [embedding.values for embedding in result.embeddings]
        return embeddings

gemini_embedding_function = GeminiEmbeddingFunction()

# Create collections if they don't exist
chroma_memory_self = chroma.get_or_create_collection(name="memory_self", embedding_function=gemini_embedding_function)
chroma_memory_other = chroma.get_or_create_collection(name="memory_other", embedding_function=gemini_embedding_function)




# Connect to SQLite and start APS
jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:///data/jobs.sqlite')
}

scheduler = BackgroundScheduler(jobstores=jobstores)
scheduler.start()



# Execute when online. Set custom status and send a log message.
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    log_ch = bot.get_channel(log_channel_id)
    game = discord.Game("with fire")
    await bot.change_presence(status=discord.Status.online, activity=game)
    await log_ch.send(f"Yahallo~ \nIris is up and running")
    # Log self_id if it doesn't exist
    if not config_list["id_list"]["self_id"]:
        config_list["id_list"]["self_id"] = str(bot.user.id)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_list, f, indent=4)
        print("Self_id was not set. Setting it now.")
        reload_config()




# ____COMMANDS____
# Ping command.
@bot.command(description="sends the bot's latency.")
async def ping(ctx):
    bot_ping = (bot.latency * 1000)
    ping_time = round(bot_ping)
    await ctx.respond(f'pong!\nlatency is {ping_time}ms')


# Nickname command.
@bot.command(description="sets your nickname. for testing purposes.")
async def nick(ctx, nickname: str):
    await ctx.respond("please wait...")
    auth_id = f"{ctx.author.id}"

    data = await read_data_async(config_file)

    data["name_list"][auth_id] = nickname

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    reload_config()

    await ctx.send(f"nickname set. please use the command again if you wish to change it.")


# Purge command. Experimental.
@bot.command(description="nuke the memory. use this as a last resort.")
async def purge(ctx):
    if ctx.author.id == owner_id:
        backup_dir = "backups/memory"
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_file = os.path.join(backup_dir, f"memory_{timestamp}.json")

        # backup first
        shutil.copy(chat_log_file, backup_file)

        # then reset by writing an empty array
        with open(chat_log_file, "w", encoding='utf-8') as f:
            json.dump([], f, indent=4)

        await ctx.respond("done.\ngoodbye, old iris. welcome, new iris.")

    else:
        await ctx.respond("no fucking way buddy.", ephemeral=True)


# Image processing toggle command.
@bot.command(description="Toggle the image processing function on or off.")
async def image(ctx):
    global image_processing

    image_processing = not config_list["settings"]["image_processing"]
    config_list["settings"]["image_processing"] = not config_list["settings"]["image_processing"]


    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_list, f, indent=4)

    if config_list["settings"]["image_processing"]:
        await ctx.respond(f'Done.\nImage processing has been enabled.')
    if not config_list["settings"]["image_processing"]:
        await ctx.respond(f'Done.\nImage processing has been disabled.')



# Incoming message handling. Uses a list as a buffer for multiple messages.
response_task = None
message_buffer = []

@bot.event
async def on_message(message):
    attachment_bool = False
    attachments_amount = 0
    ext = None
    global response_task

    if message.author.bot or message.author == bot.user:
        return

    if message.reference and message.reference.resolved.author != bot.user:
        return

    elif (message.channel.id == message_channel or self_name in message.content.lower()
          or bot.user.mentioned_in(message)):

        if message.attachments:
            attachment_bool = True
            for attachment in message.attachments:
                if attachment.content_type.startswith("image/"):
                    if image_processing:
                        attachments_amount += 1
                        ext = os.path.splitext(attachment.filename)[1]  # Gets .jpg, .png, etc.
                        filename = f"downloads/image{ext}"

                        async with aiohttp.ClientSession() as session:
                            async with session.get(attachment.url) as resp:
                                if resp.status == 200:
                                    with open(filename, "wb") as f:
                                        f.write(await resp.read())

                    if not image_processing:
                        attachments_amount += 1



        try:
            # Get name (in case there is no id in the .json)
            name_backup_a = f"{message.author}"

            index_start = name_backup_a.find("(")
            index_end = name_backup_a.find(")")
            name_backup_b = name_backup_a[index_start + 1:index_end]


            # Get name from id (from the .json)
            auth = f"{message.author.id}"

            name = name_list.get(auth, name_backup_b)

            # Prompt preparation
            prompt = message.content
            prompt_clean = prompt.replace(f'<@{self_id}>', '')

            message_buffer.append(prompt_clean.strip())
            # print(message_buffer)

            if response_task and not response_task.done():
                response_task.cancel()

            response_task = asyncio.create_task(wait_and_respond(message, name, auth, attachment_bool, ext, attachments_amount))


        except Exception as e:
            channel = bot.get_channel(log_channel_id)
            await channel.send(f"An error occurred:\n```{str(e)}```")
            print(f"It broke\nError message:\n{e}")



# Json output structure
class Memory_Structure(BaseModel):
    memory_self: Optional['Memory_Structure_Self'] = None
    memory_other: Optional['Memory_Structure_Other'] = None

class Memory_Structure_Self(BaseModel):
    sentiment: str
    fact: str

class Memory_Structure_Other(BaseModel):
    name: str
    sentiment: str
    fact_type: Literal[
        "identity",
        "interests",
        "life_context",
        "communication_style",
        "other"
    ]
    fact: str
    timestamp_iso: str
    last_topic: str


class Reminder_Structure(BaseModel):
    event_time_iso: str
    event_type: Literal[
        "birthday",
        "holiday",
        "anniversary",
        "other",
        "one-time"] = "one-time"
    event_context: str
    event_person: str

class Event_Structure(BaseModel):
    event_time_cron: str
    event_type: Literal[
        "birthday",
        "holiday",
        "anniversary",
        "other"
    ]
    event_context: str
    event_person: str

class Response_Structure(BaseModel):
    responding_to_who: str
    message: str
    required_action: Literal[
        "none",
        "ask_user",
        "set_reminder",
        "set_reminder_recurring",
        "remove_reminder"
    ]
    reminder_details: Optional[Reminder_Structure] = None
    reminder_details_recurring: Optional[Event_Structure] = None
    emotion_vad: list[float]
    memory: Memory_Structure
    confidence_score: float


# Auxiliary
class Memory_Auxiliary(BaseModel):
    memory_type: Literal[
        "identity",
        "interests",
        "life_context",
        "communication_style",
        "other"
    ]
    query: str

class Response_Structure_Auxiliary(BaseModel):
    memory_query: Optional[Memory_Auxiliary] = None




# Response generation and returning using Gemini.
async def wait_and_respond(message, sender, sender_id, attachment, ext, attachments_amount):
    await asyncio.sleep(3)
    global use_fallback_model, last_fallback_time, fallback_timeout, fallback_timeout_overloaded, gemini_model_current, gemini_overloaded

    log_channel = bot.get_channel(log_channel_id)

    if not message_buffer:
        return


    combined_messages = "\n".join(message_buffer)

    # Get current time
    current_time = datetime.now()
    time_formatted = current_time.strftime("%a, %d/%m/%Y - %H:%M")

    chat_history = await read_data_async(chat_log_file)

    emotion_history = await read_data_async(emotion_state_file)
    memory_emotion = emotion_history.get("previous_emotion")

    # Auxiliary prompt for memory retrieval
    prompt_aux = {
        "sender": sender,
        "message": combined_messages,
        "images": attachment,
        "images_amount": attachments_amount,
        "chat_history": chat_history
    }

    prompt_aux_str = json.dumps(prompt_aux)

    response_aux = gemini.models.generate_content(
        model=gemini_model_auxiliary,
        config=types.GenerateContentConfig(
            system_instruction=instructions_auxiliary,
            temperature=1.2,
            response_mime_type="application/json",
            response_schema=Response_Structure_Auxiliary),
        contents=prompt_aux_str)


    response_aux_json = json.loads(response_aux.text)

    if response_aux_json["memory_query"]:
        query_result = chroma_memory_other.query(
            query_texts = response_aux_json["memory_query"]["query"],
            where = {"type": response_aux_json["memory_query"]["memory_type"]}
        )
    elif not response_aux_json["memory_query"]:
        query_result = chroma_memory_other.get(
            where = {"name": sender}
        )

    memories = [doc for group in query_result['documents'] for doc in group]



    prompt_main = {
        "current_time": time_formatted,
        "sender": sender,
        "message": combined_messages,
        "images": attachment,
        "images_amount": attachments_amount,
        "previous_emotional_state": memory_emotion,
        "related_memories": memories,
        "chat_history": chat_history
    }

    print(prompt_main)

    prompt_main_str = json.dumps(prompt_main)


    # Check if the cooldown is over
    if use_fallback_model and not gemini_overloaded and time.time() - last_fallback_time > fallback_timeout:
        use_fallback_model = False
        gemini_overloaded = False

    if use_fallback_model and gemini_overloaded and time.time() - last_fallback_time > fallback_timeout_overloaded:
        use_fallback_model = False
        gemini_overloaded = False


    if attachment and image_processing:
        image = gemini.files.upload(file=f"downloads/image{ext}")
    else:
        image = None

    contents = prompt_main_str if image is None else [image, prompt_main_str]

    if not use_fallback_model:
        gemini_model_current = gemini_model_main
    else:
        gemini_model_current = gemini_model_fallback

    # Attempt to prompt using the main model, or fallback to the other model
    try:
        response = generate_response(contents)

    # Fallback
    except Exception as e:
        if "429" in str(e).lower() or "resource_exhausted" in str(e).lower():
            # Log the error and switch models
            await log_channel.send(f"<@{owner_id}>\nQuota reached. Switching to fallback model.")
            await log_channel.send(f"Error details: {str(e)}")
            use_fallback_model = True
            gemini_overloaded = False
            last_fallback_time = time.time()
            gemini_model_current = gemini_model_fallback

            response = generate_response(contents)

        elif "503" in str(e).lower():
            await log_channel.send(f"<@{owner_id}>\nModel overloaded. Switching to fallback model.")
            await log_channel.send(f"Error details: {str(e)}")
            use_fallback_model = True
            gemini_overloaded = True
            last_fallback_time = time.time()
            gemini_model_current = gemini_model_fallback

            response = generate_response(contents)

        else:
            print(e)
            await log_channel.send(e)



    response_str = response.text

    await log_channel.send(response_str)
    await log_channel.send(f"--------------------------------------\nCurrent model: {gemini_model_current}\n--------------------------------------")


    response_json = json.loads(response_str)

    response_output = response_json["message"]

    response_emotion = response_json["emotion_vad"]


    # Responding
    async with message.channel.typing():
        await asyncio.sleep(3)
        await message.channel.send(response_output)


    # Embedding
    if response_json["memory"]["memory_other"]:
        print("Memory found")
        chroma_memory_other.add(
            ids = str(uuid.uuid4()),
            documents = response_json["memory"]["memory_other"]["fact"],
            metadatas = {"time": response_json["memory"]["memory_other"]["timestamp_iso"],
                         "name": response_json["memory"]["memory_other"]["name"],
                         "sentiment": response_json["memory"]["memory_other"]["sentiment"],
                         "type": response_json["memory"]["memory_other"]["fact_type"]}
        )
        print(f"Memory added: {response_json['memory']['memory_other']['fact']}")
        print(chroma_memory_other.peek())





    # Reminder handling
    if response_json["required_action"] == "set_reminder":
        schedule = response_json["reminder_details"]["event_time_iso"]
        context = response_json["reminder_details"]["event_context"]
        person = response_json["reminder_details"]["event_person"]
        reminder_time = datetime.fromisoformat(schedule)
        scheduler.add_job(
            schedule_reminder,
            trigger='date',
            run_date=reminder_time,
            kwargs={'user_id': sender_id, 'context': context, 'person': person})

        print(f"job scheduled for {reminder_time}")
        await log_channel.send(f"Reminder ```{context}``` scheduled for {reminder_time}.")


    # Reminder handling (recurring)
    if response_json["required_action"] == "set_reminder_recurring":
        event_schedule_str = response_json["reminder_details_recurring"]["event_time_cron"]
        event_person = response_json["reminder_details_recurring"]["event_person"]
        event_type = response_json["reminder_details_recurring"]["event_type"]
        event_context = response_json["reminder_details_recurring"]["event_context"]

        event_schedule = json.loads(event_schedule_str)

        event_id = str(uuid.uuid4())

        recurring_event = {
            "id": event_id,
            "user_id": sender_id,
            "name": event_person,
            "type": event_type,
            "schedule": event_schedule,
            "context": event_context
        }


        # Add directly to APScheduler with persistence
        scheduler.add_job(
            schedule_reminder_recurring,
            id=event_id,
            trigger='cron',
            **event_schedule,
            kwargs={'id': event_id,'user_id': sender_id, 'name': event_person, 'type': event_type, 'context': event_context},
            replace_existing=False
        )

        print(f"job scheduled with cron: {recurring_event}")


    # Reminder handling (delete)
    if response_json["required_action"] == "remove_reminder":
        event_person = response_json["reminder_details_recurring"]["event_person"]
        event_type = response_json["reminder_details_recurring"]["event_type"]
        event_context = response_json["reminder_details_recurring"]["event_context"]

        await remove_reminder(event_person, event_type, event_context)





    # Saving chat history
    chat_log = await read_data_async(chat_log_file)

    response_dict_input = {
        "time": time_formatted,
        "sender": sender,
        "text": combined_messages
    }

    response_dict_output = {
        "time": datetime.now().strftime("%d/%m/%Y - %H:%M"),
        "sender": self_name,
        "text": response_output
    }

    chat_log.append(response_dict_input)
    chat_log.append(response_dict_output)

    with open(chat_log_file, "w", encoding="utf-8") as f:
        json.dump(chat_log, f, indent=4)

    # Saving last emotion state
    last_emotion = {
        "previous_emotion": response_emotion
    }

    with open(emotion_state_file, "w", encoding="utf-8") as f:
        json.dump(last_emotion, f, indent=4)


    # Empty downloads folder
    folder = "downloads"

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


    message_buffer.clear()


# Response generation
def generate_response(contents):
    response = gemini.models.generate_content(
        model=gemini_model_current,
        config=types.GenerateContentConfig(
            system_instruction=instructions_complete,
            temperature=1.2,
            response_mime_type="application/json",
            response_schema=Response_Structure),
        contents=contents)
    return response


# When a reminder is triggered
async def reminder(user_id, context, person):
    log_channel = bot.get_channel(log_channel_id)
    channel = bot.get_channel(message_channel)


    emotion_history = await read_data_async(emotion_state_file)
    memory_emotion = emotion_history.get("previous_emotion")
    chat_history = await read_data_async(chat_log_file)

    prompt_main = {
        "message": f"The time is up for this reminder: {context}. This person(s):{person} is involved. Please respond accordingly. Ping the user with '<@{user_id}>' instead of calling their name. Do not acknowledge this message.",
        "previous_emotional_state": memory_emotion,
        "chat_history": chat_history
    }

    prompt_main_str = json.dumps(prompt_main)

    try:
        response = generate_response(prompt_main_str)
    except:
        # Fallback to the fallback model if the main model fails
        global use_fallback_model, last_fallback_time, gemini_model_current
        use_fallback_model = True
        last_fallback_time = time.time()
        gemini_model_current = gemini_model_fallback

        response = generate_response(prompt_main_str)

    response_str = response.text

    # Testing
    await log_channel.send(response_str)
    await log_channel.send("----------")
    await log_channel.send(f"Current model: {gemini_model_current}")
    await log_channel.send("--------------------------------------")

    response_json = json.loads(response_str)

    response_output = response_json["message"]

    async with channel.typing():
        await asyncio.sleep(3)
        await channel.send(response_output)



# When a recurring reminder is triggered
async def reminder_recurring(id, user_id, name, type=None, context=None):
    log_channel = bot.get_channel(log_channel_id)
    channel = bot.get_channel(message_channel)

    emotion_history = await read_data_async(emotion_state_file)
    memory_emotion = emotion_history.get("previous_emotion")
    chat_history = await read_data_async(chat_log_file)

    prompt_main = {
        "message": f"The time is up for this '{type}' event reminder: {context} \nThis/These people are involved in this event:{name}\n. Please respond accordingly. Ping the user with '<@{user_id}>' instead of calling their name. Do not acknowledge this message.",
        "previous_emotional_state": memory_emotion,
        "chat_history": chat_history
    }

    prompt_main_str = json.dumps(prompt_main)

    try:
        response = generate_response(prompt_main_str)
    except:
        # Fallback to the fallback model if the main model fails
        global use_fallback_model, last_fallback_time, gemini_model_current
        use_fallback_model = True
        last_fallback_time = time.time()
        gemini_model_current = gemini_model_fallback

        response = generate_response(prompt_main_str)

    response_str = response.text

    # Testing
    await log_channel.send(response_str)
    await log_channel.send("----------")
    await log_channel.send(f"Current model: {gemini_model_current}")
    await log_channel.send("--------------------------------------")

    response_json = json.loads(response_str)

    response_output = response_json["message"]

    async with channel.typing():
        await asyncio.sleep(3)
        await channel.send(response_output)


class Event_Removal(BaseModel):
    event_id: str
    user_id: int
    event_person: str
    event_context: str


# Delete reminders
async def remove_reminder(user, event_type, event_context):
    log_channel = bot.get_channel(log_channel_id)
    candidates = []
    for job in scheduler.get_jobs():
        kwargs = job.kwargs
        if (
                kwargs.get("type") == event_type
                and kwargs.get("name") == user
        ):
            candidates.append({
                "id": job.id,
                "context": kwargs.get("context"),
                "time": f'{job.trigger.fields[1]}:{job.trigger.fields[0]}',  # hour:minute
            })

    prompt_main = {
        "message": "Choose the closest match from the candidates provided using the context.",
        "reminder_user": user,
        "reminder_type": event_type,
        "provided_context": event_context,
        "potential_candidates": candidates
    }

    prompt_main_str = json.dumps(prompt_main)

    response = gemini.models.generate_content(
        model=gemini_model_auxiliary,
        config=types.GenerateContentConfig(
            system_instruction=instructions_complete,
            temperature=1.2,
            response_mime_type="application/json",
            response_schema=Event_Removal),
        contents=prompt_main_str)

    response_str = response.text

    response_json = json.loads(response_str)

    job_id = response_json["event_id"]
    user_id = response_json["user_id"]

    if job_id:
        scheduler.remove_job(job_id)
        await log_channel.send(f"Reminder with ID {job_id} for {user_id} removed.")
        return f"Reminder with ID {job_id} for {user_id} removed."
    else:
        return "Could not determine which reminder to remove."



# Create reminders (sync)
def schedule_reminder_recurring(id, user_id, name, type=None, context=None):
    bot.loop.create_task(reminder_recurring(id, user_id, name, type, context))

def schedule_reminder(user_id, context, person):
    bot.loop.create_task(reminder(user_id, context, person))


# Reload config after changes
def reload_config():
    global config_list, name_list, locations_list, id_list
    with open(config_file, "r", encoding="utf-8") as f:
        config_list = json.load(f)

    name_list = config_list["name_list"]
    locations_list = config_list["file_locations"]
    id_list = config_list["id_list"]
    print("Config reloaded.")


# Read data files (async)
async def read_data_async(path):
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content)


for job in scheduler.get_jobs():
    print(f"job id: {job.id}, trigger: {job.trigger}, next run: {job.next_run_time}")
    print(f"job kwargs: {job.kwargs}")


try:
    bot.run(os.getenv('DISCORD_BOT_TOKEN'))
except discord.errors.LoginFailure:
    print("No token or improper token was given")
