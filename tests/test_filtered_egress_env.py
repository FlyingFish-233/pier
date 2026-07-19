import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from pier.environments.agent_setup import (
    EGRESS_PROXY_PORT,
    EGRESS_PROXY_SERVICE,
    docker_run_command,
    egress_proxy_image_name,
    proxy_environment,
    write_docker_proxy_compose,
)
from pier.environments.base import ExecResult
from pier.environments.docker.docker import DockerEnvironment
from pier.environments.modal import ModalEnvironment, _ModalDinD, _ModalDirect


def test_docker_proxy_compose_does_not_inject_proxy_env_into_main(tmp_path):
    path = tmp_path / "docker-compose-egress-proxy.json"
    write_docker_proxy_compose(
        path=path,
        proxy_dir=tmp_path / "proxy",
        allowlist=type("Allowlist", (), {"domains": ["api.openai.com"]})(),
        token="secret",
    )

    compose = json.loads(path.read_text())
    main = compose["services"]["main"]
    assert "environment" not in main
    assert main["networks"] == ["pier-egress-internal"]
    assert EGRESS_PROXY_SERVICE in main["depends_on"]

    proxy = compose["services"][EGRESS_PROXY_SERVICE]
    assert proxy["image"] == egress_proxy_image_name()
    assert proxy["pull_policy"] == "never"
    assert "build" not in proxy


def test_egress_proxy_image_name_is_content_addressed():
    image_name = egress_proxy_image_name()

    assert image_name.startswith("pier-egress-proxy:")
    assert len(image_name.rsplit(":", 1)[1]) == 16


def test_docker_agent_process_env_adds_proxy_only_for_agent_commands():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._egress_proxy_env = proxy_environment(
        "secret", EGRESS_PROXY_SERVICE, EGRESS_PROXY_PORT
    )

    process_env = env.agent_process_env({"OPENAI_API_KEY": "test"})

    assert process_env["OPENAI_API_KEY"] == "test"
    assert process_env["HTTP_PROXY"].startswith("http://agent:secret@")


def test_cached_egress_proxy_image_skips_build(tmp_path):
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._egress_proxy_image_name = egress_proxy_image_name()
    env._egress_proxy_build_context_dir = tmp_path
    env._docker_image_exists = AsyncMock(return_value=True)

    with patch(
        "pier.environments.docker.docker.asyncio.create_subprocess_exec",
        new_callable=AsyncMock,
    ) as create_subprocess:
        asyncio.run(env._ensure_egress_proxy_image())

    env._docker_image_exists.assert_awaited_once_with(egress_proxy_image_name())
    create_subprocess.assert_not_awaited()


def test_docker_delete_removes_only_compose_local_images():
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._keep_containers = False
    env.logger = MagicMock()
    env.prepare_logs_for_host = AsyncMock()
    env._run_docker_compose_command = AsyncMock()
    env._cleanup_resources_compose_file = MagicMock()

    asyncio.run(env.stop(delete=True))

    env._run_docker_compose_command.assert_awaited_once_with(
        ["down", "--rmi", "local", "--volumes", "--remove-orphans"]
    )
    env._cleanup_resources_compose_file.assert_called_once_with()


def test_modal_agent_process_env_adds_proxy_only_for_agent_commands():
    env = ModalEnvironment.__new__(ModalEnvironment)
    env._egress_proxy_env = proxy_environment("secret", "r123.modal.host", 12345)

    process_env = env.agent_process_env({"OPENAI_API_KEY": "test"})

    assert process_env["OPENAI_API_KEY"] == "test"
    assert process_env["HTTP_PROXY"] == "http://agent:secret@r123.modal.host:12345"


def test_modal_direct_exec_uses_non_login_shell():
    class DummyModalEnv:
        def __init__(self):
            self.kwargs = None

        async def _sdk_exec(self, command, **kwargs):
            self.kwargs = kwargs
            return ExecResult(return_code=0)

    env = DummyModalEnv()
    result = asyncio.run(_ModalDirect(env).exec("echo ok"))

    assert result.return_code == 0
    assert "login" not in env.kwargs


def test_agent_dockerfile_install_uses_non_login_shell():
    assert docker_run_command("echo $PATH") == 'RUN ["/bin/bash", "-c", "echo $PATH"]'


def test_modal_dind_compose_exec_uses_non_login_shell():
    env = ModalEnvironment.__new__(ModalEnvironment)
    env.environment_name = "task"
    env.task_env_config = type(
        "TaskEnv",
        (),
        {"env": {}, "cpus": 1, "memory_mb": 1024, "docker_image": None},
    )()
    env._persistent_env = {}
    strategy = _ModalDinD(env)

    captured = []

    async def compose_exec(parts, timeout_sec=None):
        captured.append(parts)
        return ExecResult(return_code=0)

    strategy._compose_exec = compose_exec

    async def run():
        await strategy.exec("echo ok")

    asyncio.run(run())

    assert captured[0][-3:] == ["bash", "-c", "echo ok"]


def test_modal_exec_preserves_env_when_switching_user():
    class DummyStrategy:
        def __init__(self):
            self.command = None
            self.env = None

        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            self.command = command
            self.env = env
            return ExecResult(return_code=0)

    env = ModalEnvironment.__new__(ModalEnvironment)
    env.default_user = None
    env._persistent_env = {}
    env.task_env_config = type("TaskEnv", (), {"workdir": None})()
    env._strategy = DummyStrategy()

    result = asyncio.run(
        ModalEnvironment.exec(env, "echo $PATH", user="agent", env={"PATH": "/custom"})
    )

    assert result.return_code == 0
    assert env._strategy.env == {"PATH": "/custom"}
    assert env._strategy.command == "su -m agent -s /bin/bash -c 'echo $PATH'"
