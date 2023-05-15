from http import HTTPStatus

import pytest
import yaml
from starlette.testclient import TestClient

from service.settings import ServiceConfig

with open('./service/envs/authentication_env.yaml') as env_config:
    ENV_TOKEN = yaml.safe_load(env_config)


@pytest.fixture
def health_path() -> str:
    return "/health"


@pytest.fixture
def reco_path() -> str:
    return "/reco/{model_name}/{user_id}"


@pytest.fixture
def explain_path() -> str:
    return "/explain/{model_name}/{user_id}/{item_id}"


def test_health(
    health_path,
    client: TestClient,
) -> None:
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(health_path)
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    reco_path,
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    path = reco_path.format(model_name="als", user_id=user_id)
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(path)
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    reco_path,
    client: TestClient,
) -> None:
    user_id = 10 ** 10
    path = reco_path.format(model_name="als", user_id=user_id)
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_unknown_model(
    reco_path,
    client: TestClient,
) -> None:
    path = reco_path.format(model_name="unknown", user_id=10 ** 9)
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"


def test_get_reco_with_incorrect_token(
    reco_path,
    client: TestClient,
) -> None:
    model_name = 'als'
    user_id = 666
    path = reco_path.format(model_name=model_name, user_id=user_id)
    with client:
        response = client.get(path)
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["errors"][0]["error_key"] == "token_is_not_correct"


def test_get_explain_user_in_train_success(
    explain_path,
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    users_id = [176549, -1]
    items_id = [14961, 2788]
    for idx in range(len(users_id)):
        path = explain_path.format(model_name="als",
                                   user_id=users_id[idx],
                                   item_id=items_id[idx])
        with client:
            client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
            response = client.get(path)
        assert response.status_code == HTTPStatus.OK
        response_json = response.json()
        assert "score" in response_json
        assert "explanation" in response_json
        assert isinstance(response_json["score"], int)
        assert response_json["explanation"] != ""
        assert isinstance(response_json["explanation"], str)


def test_get_explain_for_unknown_user(
    explain_path,
    client: TestClient,
) -> None:
    path = explain_path.format(model_name="als",
                               user_id=10 ** 10,
                               item_id=14961)
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_explain_for_unknown_model(
    explain_path,
    client: TestClient,
) -> None:
    path = explain_path.format(model_name="unknown",
                               user_id=10 ** 9,
                               item_id=14961)
    with client:
        client.headers = {"Authorization": f"Bearer {ENV_TOKEN['token']}"}
        response = client.get(path)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"
