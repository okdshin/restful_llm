import argparse
import base64
import io
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
from fastapi import APIRouter, FastAPI, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from flask import Flask, escape, render_template, request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.backends.cudnn.enabled = False
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

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


class GenerateTokensParams(CommonGenerateParams):
    input_tokens: List[int]


class GenerateTextParams(CommonGenerateParams):
    input_text: str


class RestfulLLMApp:
    def __init__(self, model_name: str, tokenizer_name: Optional[str]):
        self.model_name = model_name
        self.router = APIRouter()
        self.router.add_api_route("/docs", self.docs, methods=["GET"])
        self.router.add_api_route(
            "/api/v1/default_config/", self.default_config, methods=["GET"]
        )

        self.router.add_api_route("/api/v1/tokenize/", self.tokenize, methods=["POST"])
        self.router.add_api_route(
            "/api/v1/detokenize/", self.detokenize, methods=["POST"]
        )
        self.router.add_api_route(
            "/api/v1/generate_tokens/", self.generate_tokens, methods=["POST"]
        )
        self.router.add_api_route(
            "/api/v1/generate_text/", self.generate_text, methods=["POST"]
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def docs(self):
        return RedirectResponse(url="/docs")

    def default_config(self) -> JSONResponse:
        default_config_dict = dict(
            max_length=2048,
        )
        return JSONResponse(content=jsonable_encoder(default_config_dict))

    def tokenize(self, text: Text) -> Tokens:
        tokens = self.tokenizer(text.text).input_ids
        return Tokens(tokens=tokens)

    def detokenize(self, tokens: Tokens) -> Text:
        text = self.tokenizer.decode(tokens.tokens)
        return Text(text=text)

    def generate_tokens(self, params: GenerateTokensParams) -> Tokens:
        generated_tokens = self.model.generate(
            inputs=torch.LongTensor([params.input_tokens]),
            max_length=params.max_length,
            do_sample=params.do_sample,
            top_k=params.top_k,
            top_p=params.top_p,
            temperature=params.temperature,
        )[0]
        return Tokens(tokens=generated_tokens.tolist())

    def generate_text(self, params: GenerateTextParams) -> Text:
        input_tokens: Tokens = self.tokenize(text=Text(text=params.input_text))
        params_dict = vars(params)
        params_dict.pop("input_text")
        params_dict["input_tokens"] = input_tokens.tokens
        tokens_params = GenerateTokensParams(**params_dict)
        generated_tokens = self.generate_tokens(params=tokens_params)
        return self.detokenize(tokens=generated_tokens)


flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return render_template("index.html", title="RestfulLLM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help=":")
    parser.add_argument("--port", type=int, default=8000, help=":")
    parser.add_argument("--model-name", type=str, default="gpt2", help=":")
    parser.add_argument("--tokenizer-name", type=str, help=":")
    args = parser.parse_args()
    print(args)

    print(f'load from "{args.model_name}"')
    restful_llm_app = RestfulLLMApp(
        tokenizer_name=args.tokenizer_name, model_name=args.model_name
    )
    fast_api_app = FastAPI()
    fast_api_app.include_router(restful_llm_app.router)
    fast_api_app.mount("/", WSGIMiddleware(flask_app))

    import uvicorn

    uvicorn.run(fast_api_app, host=args.host, port=args.port, workers=1)
