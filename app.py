import asyncio
import io
import json
import os
from typing import List, Dict, Any

import aiohttp
import requests
from PIL import Image
from aiohttp import web

from models.inference import CaptionService

telegram_token = os.environ["TELEGRAM_TOKEN"]
api_url = f"https://api.telegram.org/bot{telegram_token}"
file_api_url = f"https://api.telegram.org/file/bot{telegram_token}"
pytorch_model = os.environ["PYTORCH_MODEL"]
word_map_file = os.environ['WORD_MAP_FILE']

caption_service = CaptionService(pytorch_model, word_map_file)


def _get_file_path(file_id: str) -> str:
    response = requests.get(f"{api_url}/getFile", data={"file_id": file_id}).json()
    return response["result"]["file_path"]


def _download_photo(file_id: str, file_path: str) -> bytes:
    response = requests.get(f"{file_api_url}/{file_path}", data={"file_id": file_id})
    return response.content


def _extract_json_value_by_key(json_data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    value = json_data
    try:
        for key_element in keys:
            value = value[key_element]
    except KeyError:
        return None
    return value


async def handler(request: web.Request):
    data = await request.json()
    image_metadata = _extract_json_value_by_key(data, ["message", "photo", 0])
    if image_metadata is None:
        image_metadata = _extract_json_value_by_key(data, ["message", "document"])
    if image_metadata is None:
        message = {
            'chat_id': data['message']['chat']['id'],
            'text': "Bot cannot create a caption for this message. Probably this message doesn't contain valid image"
        }
    else:
        image_file_id = image_metadata["file_id"]
        image_file_path = _get_file_path(image_file_id)
        image_content = _download_photo(image_file_id, image_file_path)
        image_content = Image.open(io.BytesIO(image_content)).convert('RGB')
        message = {
            'chat_id': data['message']['chat']['id'],
            'text': caption_service.caption(image_content)
        }

    headers = {
        'Content-Type': 'application/json'
    }

    async with aiohttp.ClientSession(loop=loop) as session:
        async with session.post(f"{api_url}/sendMessage",
                                data=json.dumps(message),
                                headers=headers) as resp:
            if resp.status != 200:
                return web.Response(status=500)
    return web.Response(status=200)


async def init_app(event_loop):
    app = web.Application(loop=event_loop, middlewares=[])
    app.router.add_post('/', handler)
    return app


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        app = loop.run_until_complete(init_app(loop))
        web.run_app(app, host='0.0.0.0', port=8443)
    except Exception as e:
        print('Error create server: %r' % e)
    finally:
        pass
    loop.close()
