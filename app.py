import argparse
import json
import io
import base64
from typing import List, Union, Optional, Tuple, Dict

from flask import render_template
from fastapi import FastAPI, APIRouter, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, RedirectResponse, StreamingResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from flask import Flask, escape, request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import torch
#torch.backends.cudnn.enabled = False
#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

print("torch cuda", torch.cuda.is_available())

import sys

class Text(BaseModel):
    text: str


class Tokens(BaseModel):
    tokens: List[int]


class CommonGenerateParams(BaseModel):
    max_length: int = 64
    do_sample: bool = False
    top_k: int = 100
    top_p: float = 0.9
    temperature: float = 0.5
    num_beams: int = 1


class GenerateTokensParams(CommonGenerateParams):
    input_tokens: List[int]


class GenerateTextParams(CommonGenerateParams):
    input_text: str


class HFTextGenerationTaskParams(BaseModel):
    inputs: str
    parameters: Optional[Dict[str, Union[bool, int, float]]]
    options: Optional[Dict[str, bool]]

"""
    top_k: int = Optional[int] = None
    top_p: float = Optional[float] = None
    temperature: float = 1.0
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = 
    max_time = 
    return_full_text: bool = True
    num_return_sequences: int = 1
    do_sample: bool = False
    num_beams: int = 2
"""


"""
# https://platform.openai.com/docs/api-reference/completions/create
class OpenAICompletionsParams(BaseModel):
    model: str
    prompt: Optional[str] = None
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, list]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of = 1
    logit_bias = Optional[Dict[int, int]] = None
    user: Optional[str] = None
"""


class RestfulLLMApp:
    def __init__(self, model_name: str, tokenizer_name: Optional[str]):
        self.model_name = model_name
        self.router = APIRouter()
        self.router.add_api_route("/docs", self.docs, methods=["GET"])
        self.router.add_api_route("/api/v1/default_config/", self.default_config, methods=["GET"])

        self.router.add_api_route("/api/v1/encode_to_tokens/", self.encode_to_tokens, methods=["POST"])
        self.router.add_api_route("/api/v1/decode_from_tokens/", self.decode_from_tokens, methods=["POST"])
        self.router.add_api_route("/api/v1/generate_tokens/", self.generate_tokens, methods=["POST"])
        self.router.add_api_route("/api/v1/generate_text/", self.generate_text, methods=["POST"])

        #self.router.add_api_route("/api/huggingface_compatible/v1/text_generation_task/", self.huggingface_text_generation_task_generate, methods=["POST"])

        #self.router.add_api_route("/api/openai_compatible/v1/completions/", self.openai_completions_generate, methods=["POST"])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def docs(self):
        return RedirectResponse(url='/docs')

    def default_config(self) -> JSONResponse:
        default_config_dict = dict(
            max_length = 2048,
        )
        return JSONResponse(content=jsonable_encoder(default_config_dict))

    def encode_to_tokens(self, text: Text) -> Tokens:
        tokens = self.tokenizer(text.text).input_ids
        return Tokens(tokens=tokens)

    def decode_from_tokens(self, tokens: Tokens) -> Text:
        text = self.tokenizer.batch_decode([tokens.tokens])[0]
        return Text(text=text)

    def generate_tokens(self, params: GenerateTokensParams) -> Tokens:
        generated_tokens = self.model.generate(
            inputs=torch.LongTensor([params.input_tokens]),
            max_length=params.max_length,
            do_sample=params.do_sample,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
            num_beams=params.num_beams,
        )
        return Tokens(tokens=generated_tokens[0].tolist())

    def generate_text(self, params: GenerateTextParams) -> Text:
        input_tokens: Tokens = self.encode_to_tokens(text=Text(text=params.input_text))
        params_dict = vars(params)
        params_dict.pop("input_text")
        params_dict["input_tokens"] = input_tokens.tokens
        tokens_params = GenerateTokensParams(**params_dict)
        generated_tokens = self.generate_tokens(params=tokens_params)
        return self.decode_from_tokens(tokens=generated_tokens)

    """
    def huggingface_text_generation_task_generate(self, params: HFTextGenerationTaskParams) -> JSONResponse:
        input_ids = self.tokenizer(params.inputs).input_ids
        if params.parameters is not None:
            parameters = dict(
                top_k = params.parameters.get("top_k"),
                top_p = params.parameters.get("top_p"),
                temperature = params.parameters.get("temperature", 1.0),
                repetition_penalty = params.parameters.get("repetition_penalty"),
                max_new_tokens = params.parameters.get("max_new_tokens"),
                max_time = params.parameters.get("max_time"),
                #return_full_text = params.parameters.get("return_full_text", True),
                num_return_sequences = params.parameters.get("num_return_sequences", 1),
                do_sample = params.parameters.get("do_sample", True),
            )
        else:
            parameters = {}
        generated_tokens = self.model.generate(
            inputs=torch.LongTensor([input_ids]),
            **parameters,
        )
        generated_text_list = self.tokenizer.batch_decode(generated_tokens)
        res = [dict(generated_text=gt) for gt in generated_text_list]
        return JSONResponse(content=jsonable_encoder(res))

    def openai_completions_generate(self, params: OpenAICompletionsParams) -> JSONResponse:
        pass
    """


flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html", title="LocalLLM")


if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", type=str, default="0.0.0.0", help=":")
    parser.add_argument("--port", type=int, default=8000, help=":")
    parser.add_argument("--model-name", type=str, default="gpt2", help=":")
    parser.add_argument("--tokenizer-name", type=str, help=":")
    args = parser.parse_args()
    print(args)

    print(f"load from \"{args.model_name}\"")
    restful_llm_app = RestfulLLMApp(tokenizer_name=args.tokenizer_name, model_name=args.model_name)
    fast_api_app = FastAPI()
    fast_api_app.include_router(restful_llm_app.router)
    fast_api_app.mount("/", WSGIMiddleware(flask_app))

    import uvicorn
    uvicorn.run(fast_api_app, host=args.host, port=args.port, workers=1)
