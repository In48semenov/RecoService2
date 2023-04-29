from typing import List

import yaml
from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import sentry_sdk
from sentry_sdk import capture_exception

from service.api.exceptions import (
    AuthenticateError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.configs.responses_cfg import example_responses
from service.log import app_logger
from service.utils.common_artifact import registered_model, explained_model
from service.models_inference.run_reco_pipeline import MainPipeline
from service.models_inference.popular.reco_popular import add_reco_popular
from service.casual_inference.explain import ModelOutputExplain


with open('./service/envs/authentication_env.yaml') as env_config:
    ENV_TOKEN = yaml.safe_load(env_config)

with open("./service/envs/sentry_env.yml") as file:
    sentry_dsn = yaml.safe_load(file)["dsn"]

sentry_sdk.init(
    dsn=sentry_dsn,
    traces_sample_rate=1.0
)

pipeline = MainPipeline()
model_output_explain = ModelOutputExplain()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class ExplainResponse(BaseModel):
    score: int
    explanation: str


router = APIRouter()
auth_scheme = HTTPBearer(auto_error=False)


async def authorization_by_token(
    token: HTTPAuthorizationCredentials = Security(auth_scheme),
):
    if token is not None and token.credentials == ENV_TOKEN['token']:
        return token.credentials
    else:
        raise AuthenticateError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health(
    token: HTTPAuthorizationCredentials = Depends(authorization_by_token)
) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=example_responses,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(authorization_by_token),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name not in registered_model:
        capture_exception(f"Model name '{model_name}' not found")
        raise ModelNotFoundError(
            error_message=f"Model name '{model_name}' not found"
        )

    if user_id > 10 ** 9:
        capture_exception(f"User {user_id} not found")
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    recs = pipeline.recommend(user_id=user_id, k_recs=k_recs)
    recs = add_reco_popular(k_recs=k_recs, curr_recs=recs)

    return RecoResponse(user_id=user_id, items=recs)


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    response_model=ExplainResponse,
)
async def explain(
    request: Request,
    model_name: str,
    user_id: int,
    item_id: int,
    token: HTTPAuthorizationCredentials = Depends(authorization_by_token),
) -> ExplainResponse:
    """
    Пользователь переходит на карточку контента, на которой нужно показать
    процент релевантности этого контента зашедшему пользователю,
    а также текстовое объяснение почему ему может понравиться этот контент.

    Args:
        request: запрос.
        model_name: название модели, для которой нужно получить объяснения.
        user_id: id пользователя, для которого нужны объяснения.
        item_id: id контента, для которого нужны объяснения.

    Return:
        Response со значением процента релевантности и текстовым объяснением,
        понятным пользователю.
        ExplainResponse:
            - "p": "процент релевантности контента item_id для пользователя
                    user_id"
            - "explanation": "текстовое объяснение почему рекомендован item_id"
    """
    app_logger.info(
        f"Request explanation for model: {model_name}, user_id: {user_id}"
    )

    if model_name not in explained_model:
        capture_exception(f"Model name '{model_name}' not found")
        raise ModelNotFoundError(
            error_message=f"Model name '{model_name}' not found"
        )

    if user_id > 10 ** 9:
        capture_exception(f"User {user_id} not found")
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    score, explanation = model_output_explain.explain(
        model_name, user_id, item_id, True
    )

    return ExplainResponse(score=score, explanation=explanation)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
