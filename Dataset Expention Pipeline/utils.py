from openai import OpenAI
from timeout_decorator import timeout
import time
import os
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import base64
import os
import mimetypes
import pandas as pd
from random import choice
from time import sleep
from tqdm import tqdm

random_seed = 42
# this utils support a single OpenAI key and multiple Gemini keys
openai_client = OpenAI(api_key = open('../Config/openai.txt', 'r', encoding='utf-8').read() ) # ../Config/openai.txt: put openai key here
gemini_client = None # ../Config/gemini.txt: put gemini keys here

def OpenAI_model_response(prompt, model_name):
    @timeout(15)
    def caller():
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    while True:
        try:
            return caller()
        except:
            print("Sleep for 60 sec.")
            time.sleep(60)


def Gemini_model_response(prompt, model_name, id, save_path, gemini_client=gemini_client):
    log_msg = ""
    log_path = "Outputs/Logs"
    file_path = f"{save_path}ex_{id}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if (
        os.path.exists(f"{save_path}ex_{id}.png")
        or os.path.exists(f"{save_path}ex_{id}.jpg")
        or os.path.exists(f"{save_path}ex_{id}.jpeg")
        or os.path.exists(f"{save_path}ex_{id}.gif")
    ):
        print(f"    [pass] File already exists at {file_path}.")
        return True
    if gemini_client is None:
        raise ValueError("Gemini client is not initialized.")

    def save_binary_file(file_name, data):
        f = open(file_name, "wb")
        f.write(data)
        f.close()

    file_extension = ".png"  # by default

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=100,
        response_modalities=[
            "image",
            "text",
        ],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="BLOCK_NONE",  # Block none
            ),
        ],
        response_mime_type="text/plain",
    )

    attempts = 0
    while attempts < 3:
        try:
            for chunk in gemini_client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    not chunk.candidates
                    or not chunk.candidates[0].content
                    or not chunk.candidates[0].content.parts
                ):
                    continue
                if chunk.candidates[0].content.parts[0].inline_data:
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    save_binary_file(f"{file_path}{file_extension}", inline_data.data)
                    print(
                        "File of mime type"
                        f" {inline_data.mime_type} saved"
                        f" to: {file_path}{file_extension}"
                    )
                else:
                    log_msg += str(chunk.text)
            break
        except ServerError as e:
            print(
                f"    [gemini] Server error occurred: {e}; sleep for 2 secs and try again, {attempts + 1}/3."
            )
            time.sleep(60)
            attempts += 1
    # If the file is successfully saved, return the file content
    if os.path.exists(f"{file_path}{file_extension}"):
        file_content = open(f"{file_path}{file_extension}", "rb").read()
        return file_content
    # If the file is not saved, return the log message only (possibly due to a moderation strategy)
    else:
        # save text log
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file_path = os.path.join(
            log_path,
            f"log_{id}_{current_time}.txt",
        )
        with open(log_file_path, "a") as log_file:
            log_file.write(log_msg)
        return log_msg


# A load balancer for Gemini apikeys, accepting args
def Gemini_load_balancer(prompt, model_name, **kwargs):
    config_file = "../Config/gemini.txt"
    gemini_keys = {}
    # Usage tracker for each day
    gemini_usage = {}
    gemini_usage_log = f"../Dataset Expention Pipeline/Outputs/Logs/log_ex_gemini_key_usage_{time.strftime('%Y-%m-%d')}.txt"

    if os.path.exists(gemini_usage_log):
        with open(gemini_usage_log, "r") as f:
            lines = f.readlines()
        for line in lines:
            if len(line.strip()) > 0:
                key, count = line.strip().split(":")
                gemini_usage[int(key)] = int(count)

    # Reading the keys from the config file
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            lines = f.readlines()
        index = 0
        for line in lines:
            if len(line.strip()) > 0:
                gemini_keys[index] = line.strip()
                if index not in gemini_usage:
                    gemini_usage[index] = 0
                index += 1

    # Load balancer
    if len(gemini_keys) == 0:
        raise ValueError("  [balancer] No Gemini keys found in the config file.")
    else:
        print(
            # f"  [balancer] Loading a balancer with {len(gemini_keys)} Gemini keys."
        )
        random_index = choice(range(len(gemini_keys)))
        # Check if the selected key has reached the maximum usage count
        MAX_USAGE_COUNT = 1500
        # Avoid infinite loop
        attempts = 0
        while random_index in gemini_usage and (
            gemini_usage[random_index] >= MAX_USAGE_COUNT
        ):
            random_index = choice(range(len(gemini_keys)))
            attempts += 1
            if attempts > 25:
                raise ValueError(
                    "All Gemini keys have reached the maximum usage count."
                )
        # Initialize the Gemini client with the selected key
        print(
            f"    [balancer] Using Gemini key at index {random_index}, which is {gemini_keys[random_index][:8]}..."
        )
        gemini_client = genai.Client(api_key=gemini_keys[random_index])
        response = Gemini_model_response(
            prompt, model_name, gemini_client=gemini_client, **kwargs
        )
        gemini_usage[random_index] += 1

        # Save the updated usage counts back to the config file
        with open(
            gemini_usage_log,
            "w",
        ) as f:
            # store the usage dict as json
            for key, count in gemini_usage.items():
                f.write(f"{key}: {count}\n")
    return response